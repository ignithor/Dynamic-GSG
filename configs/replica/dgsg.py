import os
from os.path import join as p_join


scenes = ["room0", "room1", "room2",
          "office0", "office1", "office2",
          "office3", "office4"]

primary_device="cuda:0"
seed = 0
scene_name = scenes[0]

map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

group_name = "Replica"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=100, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=False,
    whether_to_update = False,
    wandb=dict(
        entity="ICR-Lab",
        project="Dynamic-GSG",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/mnt/ssd1/paul/data/Replica", 
        gradslam_data_cfg="./configs/data/replica.yaml",
        sequence=scene_name,  
        desired_image_height=680,
        desired_image_width=1200,
        start=0,
        end=-1,
        stride=10,
        num_frames=200,
        ignore_bad = False,
        use_train_split = True,
    ),
    tracking=dict(
        modify_real_gt_poses=True, # Modify Real GT Poses for Tracking
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.5,
        use_l1=True,
        ignore_outlier_depth_loss=True,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            features=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.98, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            features=0.5,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            features=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    lang=dict(
        use_lang=True,
        detection_model="yolo", # ["groundingdino", "yolo"]
        color_book_path="./configs/scannet200.txt",
        yolo_model_path="./models/yolov8l-world.pt",
        grounding_dino_config_path="./dgsg/navigation/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint_path="./models/groundingdino_swint_ogc.pth",
        ram_model_path="./models/ram_plus_swin_large_14m.pth",
        sam_model_path='./models/sam_l.pt',
        dam_model_path='nvidia/DAM-3B', 
        dam_conv_mode="v1", 
        dam_prompt_mode="focal_prompt",
        sys_prompt_file="./configs/prompts/parsing_query.txt",
        obj_prompt_file="./configs/prompts/parsing_objects.txt",
        obj_caption_file="./configs/prompts/parsing_objects_caption.txt",
        clip_model_path='./models/open_clip_pytorch_model.bin',
        classes_file="./configs/scannet200_classes.txt",
        bg_classes=["wall", "floor", "ceiling"],
        skip_bg=True,
        mask_area_threshold=10,
        max_bbox_area_ratio=0.6,
        mask_conf_threshold=0.4,
        merge_overlap_thresh=0.7,
        merge_visual_sim_thresh=0.7,
        similarity_bias = 0.0,
        similarity_threshold = 0.55,
        update_gs_num_threshold = 500,
        update_gs_ratio_threshold = 0.9,
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        show_bg=True, # Show Background
        visualize_cams=False, # Visualize Camera Frustums and Trajectory
        viz_w=640, viz_h=480,
        viz_near=0.01, viz_far=100.0,
        view_scale=1,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
        no_clip=False, # If set, the CLIP model will not init for fast debugging.
        clip_model_path='./models/open_clip_pytorch_model.bin',
        variables_path=f"./experiments/Replica/{run_name}/variables.npz",
        keyframe_list_path=f"./experiments/Replica/{run_name}/keyframelist.pkl.gz",
    ),
)