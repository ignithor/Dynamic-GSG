import argparse
import os
import sys
import pickle
import gzip
import copy
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
            # Default tiny box
            obj['bbox'] = o3d.geometry.OrientedBoundingBox(np.array([0,0,0]), np.eye(3), np.array([0.01, 0.01, 0.01]))
            continue
            
        obj_points = all_means3D[mask]
        
        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        
        # Optional: Remove outliers to get a tighter bbox
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
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
            obj['bbox'] = o3d.geometry.OrientedBoundingBox(center, np.eye(3), np.array([0.01,0.01,0.01]))

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

def highlight_object_command(scene_data, original_colors, objects, objects_idx, clip_model, clip_tokenizer):
    print("\n" + "="*40)
    print("      OBJECT HIGHLIGHTING      ")
    print("="*40)
    command = input(">> Enter object description to highlight (e.g., 'the red chair'): ")
    
    # Reset colors to original before applying new highlight
    scene_data['colors_precomp'] = original_colors.clone()

    # 1. Find Object
    target_obj, score = get_object_by_text(objects, command, clip_model, clip_tokenizer)
    
    if target_obj is None:
        print(f"[System] No object found matching '{command}'.")
        return scene_data, False

    print(f"[System] Found: ID {target_obj['idx']} ({target_obj['caption']}) | Score: {score:.3f}")
    
    # 2. Find indices of the object
    # Ensure objects_idx is 1D
    if objects_idx.ndim > 1:
        objects_idx = objects_idx.flatten()
        
    target_indices = np.where(objects_idx == target_obj['idx'])[0]
    
    if len(target_indices) == 0:
        print("[System] Object has no points in the scene.")
        return scene_data, False
        
    # 3. Highlight in RED [1, 0, 0]
    # We modify the colors_precomp tensor directly
    red_color = torch.tensor([1.0, 0.0, 0.0], device="cuda", dtype=torch.float32)
    scene_data['colors_precomp'][target_indices] = red_color
    
    print(f"[System] Highlighted {len(target_indices)} points in RED.")
    return scene_data, True

def check_collision(target_obj, new_center, all_objects, ref_obj_idx, verbose=True):
    """
    Checks if the target object at the new position intersects with any OTHER object.
    Returns True if collision detected.
    """
    # Create a ghost bounding box at the new location
    ghost_bbox = copy.deepcopy(target_obj['bbox'])
    ghost_bbox.center = new_center
    
    # Slightly scale down the ghost box (e.g., 95%) to allow objects to "touch"
    # without triggering collision (like sitting ON a table).
    ghost_bbox.scale(0.95, ghost_bbox.center)

    for obj in all_objects:
        # Don't check collision with itself
        if obj['idx'] == target_obj['idx']:
            continue
        
        # Don't check collision with the reference object (e.g. the table we are putting it on)
        # We assume the user implies contact is okay.
        if obj['idx'] == ref_obj_idx:
            continue

        # Skip invalid bboxes
        if 'bbox' not in obj:
            continue

        # Check Intersection using Corner Approximation (Open3D doesn't have native OBB intersection)
        # 1. Check if corners of ghost_bbox are inside obj['bbox']
        ghost_corners = o3d.utility.Vector3dVector(np.asarray(ghost_bbox.get_box_points()))
        inside_ghost = obj['bbox'].get_point_indices_within_bounding_box(ghost_corners)
        
        if len(inside_ghost) > 0:
            if verbose:
                print(f"[Collision Alert] Proposed move collides with Object ID {obj['idx']} ({obj['caption']})")
            return True

        # 2. Check if corners of obj['bbox'] are inside ghost_bbox
        obj_corners = o3d.utility.Vector3dVector(np.asarray(obj['bbox'].get_box_points()))
        inside_obj = ghost_bbox.get_point_indices_within_bounding_box(obj_corners)

        if len(inside_obj) > 0:
            if verbose:
                print(f"[Collision Alert] Proposed move collides with Object ID {obj['idx']} ({obj['caption']})")
            return True

    return False

def check_room_bounds(pos, scene_min, scene_max, margin=0.1):
    """Checks if the position is within the room boundaries (scene min/max)."""
    # Check XY (Floor plan). Z (Height) is less strict but good to check.
    return (pos[0] > scene_min[0] + margin) and (pos[0] < scene_max[0] - margin) and \
           (pos[1] > scene_min[1] + margin) and (pos[1] < scene_max[1] - margin)

def find_valid_position(target_obj, start_pos, all_objects, ref_obj_idx, scene_min, scene_max, step_size=0.1, max_radius=2.0):
    """
    Searches for the nearest valid position around start_pos that doesn't collide AND stays in room.
    """
    # Helper to check both collision and bounds
    def is_valid(pos):
        in_bounds = check_room_bounds(pos, scene_min, scene_max)
        if not in_bounds: return False
        
        collides = check_collision(target_obj, pos, all_objects, ref_obj_idx, verbose=False)
        return not collides

    # First check if the initial position is valid
    if is_valid(start_pos):
        return start_pos

    print(f"[System] Collision or Out-of-Bounds detected. Searching for nearest valid space...")
    
    current_radius = step_size
    while current_radius <= max_radius:
        # Determine number of steps for this radius (approx circumference / step)
        circumference = 2 * np.pi * current_radius
        num_steps = int(circumference / step_size)
        if num_steps < 4: num_steps = 4
        
        angles = np.linspace(0, 2*np.pi, num_steps, endpoint=False)
        
        for angle in angles:
            offset_x = current_radius * np.cos(angle)
            offset_y = current_radius * np.sin(angle)
            
            candidate_pos = np.copy(start_pos)
            candidate_pos[0] += offset_x
            candidate_pos[1] += offset_y
            
            if is_valid(candidate_pos):
                print(f"[System] Valid position found at offset (dx={offset_x:.2f}, dy={offset_y:.2f})")
                return candidate_pos
        
        current_radius += step_size

    print(f"[System] Failed to find a valid position within {max_radius}m radius.")
    return None

def get_refined_object_mask(scene_data, target_obj, objects_idx):
    """
    Refines the object mask by combining the index mask with a spatial bounding box check.
    This helps capture border points that might be mislabeled as background (0).
    """
    if objects_idx.ndim > 1:
        objects_idx = objects_idx.flatten()

    target_mask = (objects_idx == target_obj['idx'])
    
    # Use geometry to catch border points (often labeled as background 0)
    all_points_np = scene_data['means3D'].detach().cpu().numpy()
    
    # Create temp Open3D cloud
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(all_points_np)
    
    # Use the object's BBox, scaled up slightly to catch fuzzy borders
    bbox = copy.deepcopy(target_obj['bbox'])
    bbox.scale(1.1, bbox.center) # Expand by 10%
    
    # Find indices inside this box
    indices_inside = bbox.get_point_indices_within_bounding_box(pcd_all.points)
    
    spatial_mask = np.zeros(len(objects_idx), dtype=bool)
    spatial_mask[indices_inside] = True
    
    # Only select points if they are the target object OR background (0)
    valid_types = (objects_idx == target_obj['idx']) | (objects_idx == 0)
    spatial_mask = spatial_mask & valid_types
    
    # Combine masks
    final_mask = target_mask | spatial_mask
    return final_mask

def perform_teleportation(scene_data, objects, objects_idx, clip_model, clip_tokenizer):
    print("\n" + "="*40)
    print("      SEMANTIC OBJECT TELEPORTATION      ")
    print("="*40)
    command = input(">> Enter command (e.g., 'put the chair on the table' or 'near the door'): ")
    
    relation = None
    if "near" in command:
        relation = "near"
        parts = command.split("near")
    elif "next to" in command:
        relation = "near"
        parts = command.split("next to")
    elif "on" in command:
        relation = "on"
        parts = command.split("on")
    else:
        print("Error: Command not understood. Use 'near', 'next to', or 'on'.")
        return scene_data, objects_idx, False

    target_text = parts[0].replace("put", "").replace("the", "").replace("move", "").strip()
    ref_text = parts[1].replace("the", "").strip()
    
    print(f"[System] Relation: {relation.upper()} | Target: '{target_text}' | Ref: '{ref_text}'")

    # 1. Find Objects
    target_obj, _ = get_object_by_text(objects, target_text, clip_model, clip_tokenizer)
    ref_obj, _ = get_object_by_text(objects, ref_text, clip_model, clip_tokenizer)

    if target_obj is None or ref_obj is None:
        print("[System] Failed to identify objects via CLIP.")
        return scene_data, objects_idx, False

    print(f"[System] Found Target: ID {target_obj['idx']} ({target_obj['caption']})")
    print(f"[System] Found Ref:    ID {ref_obj['idx']} ({ref_obj['caption']})")

    # 2. Calculate Geometry
    t_center = np.array(target_obj['center'])
    r_center = np.array(ref_obj['center'])
    
    t_min_bound = target_obj['bbox'].get_min_bound()
    t_max_bound = target_obj['bbox'].get_max_bound()
    
    t_height = t_max_bound[2] - t_min_bound[2]
    t_width_x = t_max_bound[0] - t_min_bound[0]
    t_width_y = t_max_bound[1] - t_min_bound[1]
    t_radius = max(t_width_x, t_width_y) / 2.0
    
    r_min_bound = ref_obj['bbox'].get_min_bound()
    r_max_bound = ref_obj['bbox'].get_max_bound()
    r_width_x = r_max_bound[0] - r_min_bound[0]
    r_width_y = r_max_bound[1] - r_min_bound[1]
    r_radius = max(r_width_x, r_width_y) / 2.0

    new_pos = np.copy(r_center)

    if relation == "near":
        # --- NEAR LOGIC ---
        direction_vec = t_center - r_center
        direction_vec[2] = 0 
        
        dist = np.linalg.norm(direction_vec)
        if dist < 0.1: direction_vec = np.array([1.0, 0.0, 0.0]) # Default X axis
        else: direction_vec = direction_vec / dist
            
        buffer_dist = 0.15 
        offset_dist = r_radius + t_radius + buffer_dist
        
        new_pos = r_center + (direction_vec * offset_dist)
        new_pos[2] = t_center[2] # Maintain original height

    elif relation == "on":
        # --- ON LOGIC ---
        new_pos[0] = r_center[0]
        new_pos[1] = r_center[1]
        
        ref_top_z = r_max_bound[2]
        new_pos[2] = ref_top_z + (t_height / 2.0)
        
        print(f"[System] Stacking Z: Ref Top {ref_top_z:.2f} + Target Half-H {t_height/2.0:.2f} = {new_pos[2]:.2f}")

    # 3. Collision & Bounds Resolution (Live Fix)
    print("[System] Validating position and checking room bounds...")
    
    # Calculate Scene Bounds (Min/Max of all points)
    all_points_tensor = scene_data['means3D']
    scene_min = torch.min(all_points_tensor, dim=0)[0].detach().cpu().numpy()
    scene_max = torch.max(all_points_tensor, dim=0)[0].detach().cpu().numpy()
    
    final_pos = find_valid_position(target_obj, new_pos, objects, ref_obj['idx'], scene_min, scene_max)
    
    if final_pos is None:
        print("[System] ABORTING: Could not find valid placement (collision or out of bounds).")
        return scene_data, objects_idx, False
    
    new_pos = final_pos

    # 4. Move the Target Gaussians using Refined Mask
    translation_vector = new_pos - t_center
    translation_tensor = torch.tensor(translation_vector, device='cuda', dtype=torch.float32)

    # Use the robust mask (same as deletion) to capture all parts of the object
    print(f"[System] Selecting points using robust geometry mask...")
    move_mask = get_refined_object_mask(scene_data, target_obj, objects_idx)
    target_indices = np.where(move_mask)[0]
    
    print(f"[System] Teleporting {len(target_indices)} points...")
    scene_data['means3D'][target_indices] += translation_tensor

    # 5. Update Indices & Scene Graph Data
    # Reassign moved points (including border points that were 0) to the target object ID
    if objects_idx.ndim > 1: objects_idx = objects_idx.flatten()
    objects_idx[target_indices] = target_obj['idx']

    target_obj['center'] = new_pos
    target_obj['bbox'].translate(translation_vector)
    target_obj['caption'] += f" ({relation} {ref_obj['caption']})"

    print("[System] Done. Refreshing view.")
    return scene_data, objects_idx, True

def perform_deletion(scene_data, objects, objects_idx, original_colors, clip_model, clip_tokenizer):
    print("\n" + "="*40)
    print("      OBJECT DELETION      ")
    print("="*40)
    command = input(">> Enter object description to delete (e.g., 'the lamp'): ")

    # 1. Find Object
    target_obj, score = get_object_by_text(objects, command, clip_model, clip_tokenizer)

    if target_obj is None:
        print(f"[System] No object found matching '{command}'.")
        return scene_data, objects, objects_idx, original_colors, False

    print(f"[System] Found: ID {target_obj['idx']} ({target_obj['caption']}) | Score: {score:.3f}")
    
    confirm = input(">> Are you sure you want to delete this object? (y/n): ")
    if confirm.lower() != 'y':
        print("[System] Deletion canceled.")
        return scene_data, objects, objects_idx, original_colors, False

    # 2. Use Refined Mask Logic
    print("[System] Refining deletion mask using geometry...")
    final_remove_mask = get_refined_object_mask(scene_data, target_obj, objects_idx)
    keep_mask = ~final_remove_mask 
    
    points_to_remove = np.sum(final_remove_mask)
    if points_to_remove == 0:
        print("[System] Object has no points in the scene (ghost object). Removing from graph only.")
    else:
        print(f"[System] Deleting {points_to_remove} Gaussian points (Index + Spatial)...")

    # 3. Filter Scene Data (Tensors)
    keep_mask_tensor = torch.from_numpy(keep_mask).to("cuda")
    
    for key in scene_data:
        if isinstance(scene_data[key], torch.Tensor):
            scene_data[key] = scene_data[key][keep_mask_tensor]
            
    # Also update the backup colors so highlighting doesn't crash later
    original_colors = original_colors[keep_mask_tensor]

    # 4. Filter Index Array (Numpy)
    if objects_idx.ndim > 1:
        objects_idx = objects_idx.flatten()
    objects_idx = objects_idx[keep_mask]

    # 5. Remove from Scene Graph (List)
    try:
        objects.remove(target_obj)
        print("[System] Object removed from Scene Graph.")
    except ValueError:
        print("[System] Warning: Object not found in list object list.")

    return scene_data, objects, objects_idx, original_colors, True

def rotate_camera_y(vis, direction):
    """
    Rotates the camera around the world Y axis (Yaw, assuming Y is Up).
    direction: +1 for Left, -1 for Right
    """
    step = 5.0 * direction # Degrees
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    
    # 1. Get W2C
    w2c = np.array(param.extrinsic)
    
    # 2. Convert to C2W
    c2w = np.linalg.inv(w2c)
    
    # 3. Create Rotation Matrix around Y
    angle = np.radians(step)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Y-axis rotation matrix (4x4)
    R_y = np.array([
        [cos_a, 0, sin_a, 0],
        [0,     1, 0,     0],
        [-sin_a, 0, cos_a, 0],
        [0,     0, 0,     1]
    ])
    
    # 4. Apply rotation (Orbit around world origin)
    new_c2w = R_y @ c2w
    
    # 5. Convert back to W2C
    new_w2c = np.linalg.inv(new_c2w)
    
    param.extrinsic = new_w2c
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_renderer()

def rotate_camera_x(vis, direction):
    """
    Rotates the camera around the world X axis (Orbit/Pitch).
    direction: +1 for Up, -1 for Down
    """
    step = 5.0 * direction # Degrees
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    
    # 1. Get W2C
    w2c = np.array(param.extrinsic)
    
    # 2. Convert to C2W
    c2w = np.linalg.inv(w2c)
    
    # 3. Create Rotation Matrix around X
    angle = np.radians(step)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # X-axis rotation matrix (4x4)
    R_x = np.array([
        [1,     0,      0, 0],
        [0, cos_a, -sin_a, 0],
        [0, sin_a,  cos_a, 0],
        [0,     0,      0, 1]
    ])
    
    # 4. Apply rotation (Orbit around world origin)
    new_c2w = R_x @ c2w
    
    # 5. Convert back to W2C
    new_w2c = np.linalg.inv(new_c2w)
    
    param.extrinsic = new_w2c
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_renderer()

# ==================================================================================
#                                   RENDERING & VIZ
# ==================================================================================

def load_scene_data(scene_path, first_frame_w2c):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    objects_idx = all_params['object_idx']
    
    # Ensure objects_idx is 1D
    if isinstance(objects_idx, np.ndarray) and objects_idx.ndim > 1:
        objects_idx = objects_idx.flatten()
    
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
    
    print("Initializing CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", viz_cfg.get('clip_model_path', None)
    )
    clip_model = clip_model.to('cuda')
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    print("Loading Scene Data...")
    dummy_w2c = torch.eye(4).cuda() 
    scene_data, objects_idx, w2c, intrinsics = load_scene_data(scene_path, dummy_w2c)
    objects = load_objects(objects_path)
    
    k = intrinsics[:3, :3]
    orig_w = 1200 
    orig_h = 680
    view_scale = viz_cfg.get('view_scale', 1.0)
    k[0, :] *= (viz_cfg['viz_w'] * view_scale) / orig_w
    k[1, :] *= (viz_cfg['viz_h'] * view_scale) / orig_h

    # Store original colors for highlighting reset
    original_colors = scene_data['colors_precomp'].clone()

    preprocess_object_geometry(scene_data, objects, objects_idx)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    
    vis.create_window(
        width=int(viz_cfg['viz_w'] * view_scale), 
        height=int(viz_cfg['viz_h'] * view_scale)
    )
    
    im, depth = render(w2c, k, scene_data, viz_cfg)
    pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = pts
    pcd.colors = cols
    vis.add_geometry(pcd)
    
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = k
    cparams.intrinsic.height = int(viz_cfg['viz_h'] * view_scale)
    cparams.intrinsic.width = int(viz_cfg['viz_w'] * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    # --- CALLBACKS ---

    def teleport_mode(vis):
        nonlocal scene_data, objects_idx
        
        scene_data, objects_idx, updated = perform_teleportation(
            scene_data, objects, objects_idx, clip_model, clip_tokenizer
        )
        
        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            curr_w2c = cam_params.extrinsic
            curr_k = cam_params.intrinsic.intrinsic_matrix.copy()
            curr_k[2, 2] = 1
            
            im, depth = render(curr_w2c, curr_k, scene_data, viz_cfg)
            new_pts, new_cols = rgbd2pcd(im, depth, curr_w2c, curr_k, viz_cfg)
            
            pcd.points = new_pts
            pcd.colors = new_cols
            vis.update_geometry(pcd)
            
            if not vis.poll_events():
                break
            vis.update_renderer()

    def highlight_mode(vis):
        nonlocal scene_data, objects_idx
        
        scene_data, updated = highlight_object_command(
            scene_data, original_colors, objects, objects_idx, clip_model, clip_tokenizer
        )
        
        if updated:
            while True:
                cam_params = view_control.convert_to_pinhole_camera_parameters()
                curr_w2c = cam_params.extrinsic
                curr_k = cam_params.intrinsic.intrinsic_matrix.copy()
                curr_k[2, 2] = 1
                
                im, depth = render(curr_w2c, curr_k, scene_data, viz_cfg)
                new_pts, new_cols = rgbd2pcd(im, depth, curr_w2c, curr_k, viz_cfg)
                
                pcd.points = new_pts
                pcd.colors = new_cols
                vis.update_geometry(pcd)
                
                if not vis.poll_events():
                    break
                vis.update_renderer()

    def delete_mode(vis):
        nonlocal scene_data, objects_idx, original_colors, objects
        
        scene_data, objects, objects_idx, original_colors, updated = perform_deletion(
            scene_data, objects, objects_idx, original_colors, clip_model, clip_tokenizer
        )
        
        if updated:
            while True:
                cam_params = view_control.convert_to_pinhole_camera_parameters()
                curr_w2c = cam_params.extrinsic
                curr_k = cam_params.intrinsic.intrinsic_matrix.copy()
                curr_k[2, 2] = 1
                
                im, depth = render(curr_w2c, curr_k, scene_data, viz_cfg)
                new_pts, new_cols = rgbd2pcd(im, depth, curr_w2c, curr_k, viz_cfg)
                
                pcd.points = new_pts
                pcd.colors = new_cols
                vis.update_geometry(pcd)
                
                if not vis.poll_events():
                    break
                vis.update_renderer()

    # --- REGISTER KEYS ---
    vis.register_key_callback(ord("M"), teleport_mode)
    vis.register_key_callback(ord("H"), highlight_mode)
    vis.register_key_callback(ord("D"), delete_mode)
    
    # Camera Rotation
    vis.register_key_callback(ord(","), lambda v: rotate_camera_y(v, 1))  # Left
    vis.register_key_callback(ord("."), lambda v: rotate_camera_y(v, -1)) # Right
    vis.register_key_callback(ord("["), lambda v: rotate_camera_x(v, 1))  # Up
    vis.register_key_callback(ord("]"), lambda v: rotate_camera_x(v, -1)) # Down

    print("\nControls:")
    print(" [M] : Trigger Object Teleportation")
    print(" [H] : Highlight Object")
    print(" [D] : Delete Object")
    print(" [,] : Rotate Camera Left (Y-Axis)")
    print(" [.] : Rotate Camera Right (Y-Axis)")
    print(" [[] : Rotate Camera Up (X-Axis)")
    print(" []] : Rotate Camera Down (X-Axis)")
    print(" [Q] : Quit")
    
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        curr_w2c = cam_params.extrinsic
        curr_k = cam_params.intrinsic.intrinsic_matrix.copy()
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

    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    config = experiment.config
    
    if "scene_path" not in config:
        results_dir = os.path.join(config["workdir"], config["run_name"])
        scene_path = os.path.join(results_dir, "params_with_idx.npz")
        objects_path = os.path.join(results_dir, "objects.pkl.gz")
    else:
        scene_path = config["scene_path"]
        objects_path = config["objects_path"]

    viz_cfg = config["viz"]
    
    main(scene_path, objects_path, viz_cfg)