import argparse
import os
import sys
import pickle
import gzip
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import open_clip
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

# --- Project Specific Imports (Assumed to exist based on your environment) ---
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation
from utils.slam_classes import MapObjectList

# ==================================================================================
#                                   CORE LOGIC
# ==================================================================================

def preprocess_object_geometry(scene_data, objects, objects_idx):
    """
    Computes the Oriented Bounding Box (OBB) and Center for all objects
    based on the loaded Gaussian points. This populates 'center' and 'bbox' keys.
    """
    print("[System] Pre-computing object geometry (centers & bounding boxes)...")
    
    # Convert all points to numpy once for faster processing
    all_means3D = scene_data['means3D'].detach().cpu().numpy()
    
    # Ensure objects_idx is numpy and 1D
    if isinstance(objects_idx, torch.Tensor):
        objects_idx = objects_idx.cpu().numpy()
    
    # Flatten if it's (N, 1) to avoid masking errors
    if objects_idx.ndim > 1:
        objects_idx = objects_idx.flatten()

    count = 0
    for obj in objects:
        idx = obj['idx']
        
        # Find points belonging to this object
        mask = (objects_idx == idx)
        
        # Skip if object has no points (or very few)
        if not np.any(mask) or np.sum(mask) < 10:
            # Set defaults to avoid KeyErrors later
            obj['center'] = np.array([0, 0, 0])
            obj['bbox'] = o3d.geometry.OrientedBoundingBox(np.array([0,0,0]), np.eye(3), np.array([0.1, 0.1, 0.1]))
            continue
            
        obj_points = all_means3D[mask]
        
        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        
        # Optional: Remove outliers to get a tighter bbox
        # nb_neighbors=20, std_ratio=2.0 is a standard conservative setting
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # If outlier removal deletes everything, revert to original
        if len(cl.points) > 0:
            pcd = cl
            
        # Compute Oriented Bounding Box
        try:
            bbox = pcd.get_oriented_bounding_box()
            bbox.color = (0, 1, 0) # Green for debug
            obj['bbox'] = bbox
            obj['center'] = bbox.center
            count += 1
        except Exception as e:
            print(f"Warning: Failed to compute bbox for object {idx}: {e}")
            center = np.mean(obj_points, axis=0)
            obj['center'] = center
            obj['bbox'] = o3d.geometry.OrientedBoundingBox(center, np.eye(3), np.array([0.1,0.1,0.1]))

    print(f"[System] Computed geometry for {count} objects.")

def get_object_by_text(objects, text_query, clip_model, clip_tokenizer):
    """Finds the object index with the highest CLIP similarity."""
    text_queries = [text_query]
    text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
    
    with torch.no_grad():
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        # Stack object CLIP features
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft").to("cuda")

        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
    
    best_idx = torch.argmax(similarities)
    score = similarities[best_idx]
    
    # Threshold to avoid selecting random noise
    if score < 0.2:
        return None, score
        
    return objects[best_idx], score

def clone_background_fill(scene_data, objects_idx, hole_center, hole_radius, fill_count=5000):
    """
    Inpainting Logic: Finds 'background' (idx 0) Gaussians near the object 
    and clones them into the hole location with random noise.
    """
    # Ensure objects_idx is 1D for safety
    if objects_idx.ndim > 1:
        objects_idx = objects_idx.flatten()

    # 1. Find background points (idx == 0 is usually background/unassigned)
    bg_indices = np.where(objects_idx == 0)[0]
    if len(bg_indices) == 0: 
        return scene_data, 0, None
    
    bg_means = scene_data['means3D'][bg_indices]
    hole_center_tensor = torch.tensor(hole_center, device='cuda', dtype=torch.float32)
    
    # 2. Find background points physically close to the hole to sample texture from
    dists = torch.norm(bg_means - hole_center_tensor, dim=1)
    nearby_bg_mask = dists < (hole_radius * 2.5) # Look slightly outside the object
    
    if nearby_bg_mask.sum() < 100:
        # Fallback: Random sample if no close neighbors
        source_indices = np.random.choice(bg_indices, size=fill_count, replace=True)
    else:
        # Sample from nearby valid background
        valid_indices = bg_indices[nearby_bg_mask.cpu().numpy()]
        source_indices = np.random.choice(valid_indices, size=fill_count, replace=True)

    # 3. Clone Parameters
    new_means = scene_data['means3D'][source_indices].clone()
    new_colors = scene_data['colors_precomp'][source_indices].clone()
    new_opacities = scene_data['opacities'][source_indices].clone()
    new_scales = scene_data['scales'][source_indices].clone()
    new_rots = scene_data['rotations'][source_indices].clone()
    
    # 4. Scramble positions to fill the hole (Uniform distribution inside the radius)
    noise = (torch.rand_like(new_means) * 2 - 1) * hole_radius
    # Flatten Z noise to ensure we fill the floor, not the air (Assumes Z is Up)
    noise[:, 2] *= 0.05 
    
    new_means = hole_center_tensor + noise

    # 5. Update Scene Data
    scene_data['means3D'] = torch.cat([scene_data['means3D'], new_means], dim=0)
    scene_data['colors_precomp'] = torch.cat([scene_data['colors_precomp'], new_colors], dim=0)
    scene_data['opacities'] = torch.cat([scene_data['opacities'], new_opacities], dim=0)
    scene_data['scales'] = torch.cat([scene_data['scales'], new_scales], dim=0)
    scene_data['rotations'] = torch.cat([scene_data['rotations'], new_rots], dim=0)
    
    # Return count to update indices
    return scene_data, len(source_indices)

def perform_teleportation(scene_data, objects, objects_idx, clip_model, clip_tokenizer):
    print("\n" + "="*40)
    print("      SEMANTIC OBJECT TELEPORTATION      ")
    print("="*40)
    command = input(">> Enter command (e.g., 'put the chair near the table'): ")
    
    # 1. Simple Parsing
    if "near" in command:
        parts = command.split("near")
    elif "next to" in command:
        parts = command.split("next to")
    else:
        print("Error: Command must contain 'near' or 'next to'.")
        return scene_data, objects_idx, False

    target_text = parts[0].replace("put", "").replace("the", "").replace("move", "").strip()
    ref_text = parts[1].replace("the", "").strip()
    
    print(f"[System] Looking for Target: '{target_text}' | Reference: '{ref_text}'")

    # 2. Find Objects
    target_obj, _ = get_object_by_text(objects, target_text, clip_model, clip_tokenizer)
    ref_obj, _ = get_object_by_text(objects, ref_text, clip_model, clip_tokenizer)

    if target_obj is None or ref_obj is None:
        print("[System] Failed to identify objects.")
        return scene_data, objects_idx, False

    print(f"[System] Found Target: ID {target_obj['idx']} ({target_obj['caption']})")
    print(f"[System] Found Ref:    ID {ref_obj['idx']} ({ref_obj['caption']})")

    # 3. Calculate Geometry
    # Ensure centers are valid (handled by preprocess_object_geometry now)
    t_center = np.array(target_obj['center'])
    r_center = np.array(ref_obj['center'])
    
    # Estimate radius from bounding box
    t_extent = target_obj['bbox'].get_max_bound() - target_obj['bbox'].get_min_bound()
    r_extent = ref_obj['bbox'].get_max_bound() - ref_obj['bbox'].get_min_bound()
    
    t_radius = max(t_extent[0], t_extent[1]) / 2.0
    r_radius = max(r_extent[0], r_extent[1]) / 2.0
    
    # Direction Vector (Reference -> Target)
    direction_vec = t_center - r_center
    direction_vec[2] = 0 # Flatten to ground
    
    dist = np.linalg.norm(direction_vec)
    if dist < 0.1: direction_vec = np.array([1.0, 0.0, 0.0]) # Default X if overlapping
    else: direction_vec = direction_vec / dist
        
    # Calculate New Position
    buffer_dist = 0.15 
    offset_dist = r_radius + t_radius + buffer_dist
    new_pos = r_center + (direction_vec * offset_dist)
    new_pos[2] = t_center[2] # Maintain original height
    
    translation_vector = new_pos - t_center
    translation_tensor = torch.tensor(translation_vector, device='cuda', dtype=torch.float32)

    # 4. Inpaint the Hole (Healing)
    print("[System] Inpainting background hole...")
    scene_data, filled_count = clone_background_fill(
        scene_data, 
        objects_idx, 
        hole_center=t_center, 
        hole_radius=t_radius
    )
    
    # Update indices array to account for new in-painted points (assign them to background 0)
    if filled_count > 0:
        new_indices = np.zeros(filled_count, dtype=objects_idx.dtype)
        # Flatten objects_idx before concatenation to ensure compatibility
        if objects_idx.ndim > 1:
            objects_idx = objects_idx.flatten()
        objects_idx = np.concatenate([objects_idx, new_indices])

    # 5. Move the Target Gaussians
    print(f"[System] Teleporting object {target_obj['idx']}...")
    target_indices = np.where(objects_idx == target_obj['idx'])[0]
    scene_data['means3D'][target_indices] += translation_tensor

    # 6. Update Scene Graph Data
    target_obj['center'] = new_pos
    target_obj['bbox'].translate(translation_vector)
    target_obj['caption'] += f" (Moved)"

    print("[System] Done. Refreshing view.")
    return scene_data, objects_idx, True

# ==================================================================================
#                                   RENDERING & VIZ
# ==================================================================================

def load_scene_data(scene_path, first_frame_w2c):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    objects_idx = all_params['object_idx']
    
    # Fix: Ensure objects_idx is 1D to prevent indexing errors later
    if isinstance(objects_idx, np.ndarray) and objects_idx.ndim > 1:
        objects_idx = objects_idx.flatten()
    
    # Extract only necessary tensors
    params = {}
    keys_to_load = ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']
    
    for k in keys_to_load:
        params[k] = torch.tensor(all_params[k]).cuda().float()

    if params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    
    # Load camera intrinsics/extrinsics
    w2c = all_params['w2c']
    intrinsics = all_params['intrinsics']
    
    return rendervar, objects_idx, w2c, intrinsics

def load_objects(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    objects = MapObjectList()
    objects.load_serializable(data)
    return objects

def render(w2c, k, scene_data, viz_cfg):
    with torch.no_grad():
        cam = setup_camera(viz_cfg['viz_w'], viz_cfg['viz_h'], k, w2c, viz_cfg['viz_near'], viz_cfg['viz_far'])
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered
        )
        im, _, depth, = Renderer(raster_settings=white_bg_cam)(**scene_data)
        return im, depth

def rgbd2pcd(color, depth, w2c, intrinsics, viz_cfg):
    width, height = viz_cfg['viz_w'], viz_cfg['viz_h']
    CX, CY = intrinsics[0][2], intrinsics[1][2]
    FX, FY = intrinsics[0][0], intrinsics[1][1]

    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

# ==================================================================================
#                                   MAIN LOOP
# ==================================================================================

def main(scene_path, objects_path, viz_cfg):
    
    # 1. Initialize CLIP
    print("Initializing CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", viz_cfg.get('clip_model_path', None)
    )
    clip_model = clip_model.to('cuda')
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    # 2. Load Data
    print("Loading Scene Data...")
    # Placeholder for first frame w2c, logic extracted from helper
    dummy_w2c = torch.eye(4).cuda() 
    scene_data, objects_idx, w2c, intrinsics = load_scene_data(scene_path, dummy_w2c)
    objects = load_objects(objects_path)
    
    # Scale intrinsics for visualization
    k = intrinsics[:3, :3]
    orig_w = 1200 # Assumptions if not in file
    orig_h = 680
    view_scale = viz_cfg.get('view_scale', 1.0)
    k[0, :] *= (viz_cfg['viz_w'] * view_scale) / orig_w
    k[1, :] *= (viz_cfg['viz_h'] * view_scale) / orig_h

    # --- Preprocess Object Geometry (Fix for KeyError: 'center') ---
    preprocess_object_geometry(scene_data, objects, objects_idx)

    # 3. Setup Open3D
    vis = o3d.visualization.VisualizerWithKeyCallback()
    
    # Adjust width/height by view_scale if your config uses it
    vis.create_window(
        width=int(viz_cfg['viz_w'] * view_scale), 
        height=int(viz_cfg['viz_h'] * view_scale)
    )
    
    # Initial Render
    im, depth = render(w2c, k, scene_data, viz_cfg)
    pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = pts
    pcd.colors = cols
    vis.add_geometry(pcd)
    
    # Setup Camera Control
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = k
    cparams.intrinsic.height = int(viz_cfg['viz_h'] * view_scale)
    cparams.intrinsic.width = int(viz_cfg['viz_w'] * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    # --- CALLBACK DEFINITION (REPRODUCING "OTHER CODE" METHOD) ---
    # The callback now contains the input() AND the rendering loop.
    def teleport_mode(vis):
        nonlocal scene_data, objects_idx
        
        # 1. Ask input and update data (Blocks GUI temporarily)
        scene_data, objects_idx, updated = perform_teleportation(
            scene_data, objects, objects_idx, clip_model, clip_tokenizer
        )
        
        # 2. Enter dedicated loop for this mode
        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            curr_w2c = cam_params.extrinsic
            curr_k = cam_params.intrinsic.intrinsic_matrix.copy()  # <--- copy() added
            
            # Since view control changes K, we need to ensure depth 1 at K[2,2]
            curr_k[2, 2] = 1
            
            # Re-render
            im, depth = render(curr_w2c, curr_k, scene_data, viz_cfg)
            new_pts, new_cols = rgbd2pcd(im, depth, curr_w2c, curr_k, viz_cfg)
            
            pcd.points = new_pts
            pcd.colors = new_cols
            vis.update_geometry(pcd)
            
            if not vis.poll_events():
                break
            vis.update_renderer()

    # Register 'M' for Move/Manipulate
    vis.register_key_callback(ord("M"), teleport_mode)

    print("\nControls:")
    print(" [M] : Trigger Object Teleportation (Input command in terminal)")
    print(" [Q] : Quit")
    
    # --- MAIN RENDER LOOP ---
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        curr_w2c = cam_params.extrinsic
        curr_k = cam_params.intrinsic.intrinsic_matrix.copy()  # <--- copy() added
        curr_k[2, 2] = 1
        
        im, depth = render(curr_w2c, curr_k, scene_data, viz_cfg)
        new_pts, new_cols = rgbd2pcd(im, depth, curr_w2c, curr_k, viz_cfg)
        pcd.points = new_pts
        pcd.colors = new_cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    # Load Config
    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    config = experiment.config
    
    # Resolve Paths
    if "scene_path" not in config:
        results_dir = os.path.join(config["workdir"], config["run_name"])
        scene_path = os.path.join(results_dir, "params_with_idx.npz")
        objects_path = os.path.join(results_dir, "objects.pkl.gz")
    else:
        scene_path = config["scene_path"]
        objects_path = config["objects_path"] # Assuming this exists in config if scene_path does

    viz_cfg = config["viz"]
    
    main(scene_path, objects_path, viz_cfg)