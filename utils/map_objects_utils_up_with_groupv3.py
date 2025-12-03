from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F 
import numpy as np
import supervision as sv
import cv2
import logging
import faiss
import pickle
import gzip
import os
import open3d as o3d
import base64
import json

from dam import DescribeAnythingModel
from tqdm import tqdm
from PIL import Image
from collections import Counter
from pathlib import Path
from scipy.ndimage import binary_erosion
from ultralytics import YOLO, SAM
from utils.slam_external import build_rotation
from utils.slam_classes import MapObjectList, DetectionList
from utils.slam_helpers import (
    transform_to_frame,
    transformed_params_for_detection
)
# from fast_gauss import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms, ops
from ram import inference_ram as inference


def read_color_book(color_book_path: str):
    color_book = []

    with open(color_book_path, 'r') as file:
        for line in file.readlines()[1:]:  # 跳过表头
            r, g, b = map(float, line.strip().split())
            color_book.append([r/255, g/255, b/255])

    return color_book


def compute_clip_features_batched(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    
    image = Image.fromarray(image)
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    preprocessed_images = []
    text_tokens = []
    
    # Prepare data for batch processing
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        class_id = detections.class_id[idx]
        text_tokens.append(classes[class_id])
        image_crops.append(cropped_image)
    
    # Convert lists to batches  B 3 224 224
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)   
    text_tokens_batch = clip_tokenizer(text_tokens).to(device)
    
    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # text_features = clip_model.encode_text(text_tokens_batch)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    # text_feats = text_features.cpu().numpy()
    # image_feats = []
    text_feats = []
    
    return image_crops, image_feats, text_feats


def resize_gobs(gobs, image):

    # If the shapes are the same, no resizing is necessary
    if gobs['mask'].shape[1:] == image.shape[:2]:
        return gobs

    new_masks = []

    for mask_idx in range(len(gobs['xyxy'])):

        mask = gobs['mask'][mask_idx]
        # Rescale the xyxy coordinates to the image shape
        x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
        x1 = round(x1 * image.shape[1] / mask.shape[1])
        y1 = round(y1 * image.shape[0] / mask.shape[0])
        x2 = round(x2 * image.shape[1] / mask.shape[1])
        y2 = round(y2 * image.shape[0] / mask.shape[0])
        gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]

        # Reshape the mask to the image shape
        mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        new_masks.append(mask)

    if len(new_masks) > 0:
        gobs['mask'] = np.asarray(new_masks)

    return gobs


def filter_gobs(
    gobs: dict,
    image: np.ndarray,
    skip_bg: bool = None,  # Explicitly passing skip_bg
    BG_CLASSES: list = None,  # Explicitly passing BG_CLASSES
    mask_area_threshold: float = 10,  # Default value as fallback
    max_bbox_area_ratio: float = None,  # Explicitly passing max_bbox_area_ratio
    mask_conf_threshold: float = None,  # Explicitly passing mask_conf_threshold
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs

    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]

        # Skip masks that are too small
        mask_area = gobs['mask'][mask_idx].sum()
        if mask_area < max(mask_area_threshold, 10):
            logging.debug(f"Skipped due to small mask area ({mask_area} pixels) - Class: {class_name}")
            continue

        # Skip the BG classes
        if skip_bg and class_name in BG_CLASSES:
            logging.debug(f"Skipped background class: {class_name}")
            continue

        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if max_bbox_area_ratio is not None and bbox_area > max_bbox_area_ratio * image_area:
                logging.debug(f"Skipped due to large bounding box area ratio - Class: {class_name}, Area Ratio: {bbox_area/image_area:.4f}")
                continue

        # Skip masks with low confidence
        if mask_conf_threshold is not None and gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < mask_conf_threshold:
                logging.debug(f"Skipped due to low confidence ({gobs['confidence'][mask_idx]}) - Class: {class_name}")
                continue

        idx_to_keep.append(mask_idx)

    # for key in gobs.keys():
    #     print(key, type(gobs[key]), len(gobs[key]))

    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes":  # Captions
            continue
        if k in ['labels', 'edges', 'detection_class_labels', 'text_feats']:
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")

    return gobs


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    '''
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.
     
    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2
        
    Returns:
        mask_sub: (N, H, W), binary mask
    '''
    N = xyxy.shape[0] # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)
    
    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)  交集不为负数

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)
    
    inter_over_box1 = inter_areas / areas[:, None] # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T # (N, N)
    
    # if the intersection area is smaller than th2 of the area of box1, 
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1) # (N, N)
    contained_idx = contained.nonzero() # (num_contained, 2)

    mask_sub = mask.copy() # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub

def mask_erosion(masks: np.ndarray):
    num = masks.shape[0]

    for i in range(num):
        masks[i, :, :] = binary_erosion(masks[i, :, :], structure=np.ones((2, 2)))

    return masks


def merge_first_objects_itself(objects: MapObjectList, cfg):
    '''
    TODO:
        完善对mapobject的自身合并
    '''
    # print("Before merging:", len(objects))
    len_a = len(objects)
    iou_similarities = compute_mask_iou_similarities(objects, objects)
    overlap_matrix = np.zeros((len_a, len_a))

    for idx_a in range(len_a):
        for idx_b in range(len_a):
            if idx_a == idx_b:
                continue

            if iou_similarities[idx_a, idx_b] < 1e-6:
                continue

            overlap_matrix[idx_a, idx_b] = iou_similarities[idx_a, idx_b]

    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    kept_objects = np.ones(len(objects), dtype=bool)

    # Sort indices of overlap ratios in descending order
    sort = np.argsort(overlap_ratio)[::-1]  
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]
    
    for i, j, ratio in zip(x, y, overlap_ratio):
        if ratio > cfg['merge_overlap_thresh']:
            visual_sim = F.cosine_similarity(torch.from_numpy(objects[i]["clip_ft"]), 
                                             torch.from_numpy(objects[j]["clip_ft"]), 
                                             dim=0)

            if (visual_sim > cfg['merge_visual_sim_thresh']) :
                if kept_objects[j]:  # Check if the target object has not been merged into another
                    # Merge object i into object j
                    objects[j] = merge_obj2_into_obj1(
                        objects[j],
                        objects[i],
                    )
                    kept_objects[i] = False  # Mark object i as 'merged'

        else:
            break  # Stop processing if the current overlap ratio is below the threshold

    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)

    return objects


def merge_curr_detection_itself(objects: DetectionList, cfg):
    '''
    TODO:
        完善对mapobject的自身合并
    '''
    # print("Before merging:", len(objects))
    len_a = len(objects)
    iou_similarities = compute_mask_iou_similarities(objects, objects)
    overlap_matrix = np.zeros((len_a, len_a))

    for idx_a in range(len_a):
        for idx_b in range(len_a):
            if idx_a == idx_b:
                continue

            if iou_similarities[idx_a, idx_b] < 1e-6:
                continue

            overlap_matrix[idx_a, idx_b] = iou_similarities[idx_a, idx_b]

    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    kept_objects = np.ones(len(objects), dtype=bool)

    # Sort indices of overlap ratios in descending order
    sort = np.argsort(overlap_ratio)[::-1]  
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]
    
    for i, j, ratio in zip(x, y, overlap_ratio):
        if ratio > cfg['merge_overlap_thresh']:
            visual_sim = F.cosine_similarity(torch.from_numpy(objects[i]["clip_ft"]), 
                                             torch.from_numpy(objects[j]["clip_ft"]), 
                                             dim=0)

            if (visual_sim > cfg['merge_visual_sim_thresh']) :
                if kept_objects[j]:  # Check if the target object has not been merged into another
                    # Merge object i into object j
                    objects[j] = merge_obj2_into_obj1(
                        objects[j],
                        objects[i],
                    )
                    kept_objects[i] = False  # Mark object i as 'merged'

        else:
            break  # Stop processing if the current overlap ratio is below the threshold

    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = DetectionList(new_objects)

    return objects

def encode_img(image_input):

    if isinstance(image_input, str):
        # 处理文件路径
        with open(image_input, "rb") as file:
            encoded_image = base64.b64encode(file.read()).decode('utf-8')
        return f"data:image;base64,{encoded_image}"
    
    elif isinstance(image_input, np.ndarray):
        # 处理 NumPy 数组
        # 确保图像是 uint8 类型
        if image_input.dtype != np.uint8:
            image_input = image_input.astype(np.uint8)
        
        # 自动检测并转换通道顺序 (OpenCV 使用 BGR，但网页需要 RGB)
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            # 如果是 3 通道图像，转换为 RGB 格式
            image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        
        # 编码为 PNG 格式 (可改为 JPEG 如果需要更小体积)
        success, buffer = cv2.imencode('.png', image_input)
        
        if not success:
            raise ValueError("图像编码失败")
        
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return f"data:image;base64,{encoded_image}"
    
    else:
        raise TypeError("输入必须是文件路径(str)或NumPy数组")


def initialize_first_timestep_gaussian_classes(
        color: torch.Tensor,
        detection_model,
        ram_model,
        ai_client,
        sam_predictor: SAM, 
        clip_model, 
        clip_preprocess, 
        clip_tokenizer,
        obj_classes,
        color_book,
        cfg
):
    objects = MapObjectList()
    image = color.cpu().numpy().astype(np.uint8)

    if cfg['detection_model'] == 'groundingdino':
        # img_url = encode_img(image)  # Encode the first image to Base64 format
        # completion = ai_client.chat.completions.create(
        #     model="qwen2.5-vl-72b-instruct",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        #     messages=[{"role": "user","content": [
        #             {"type": "image_url",
        #             "image_url": {"url": f"{img_url}"}},
        #             {"type": "text", "text": "Describe visible object categories in front of you. Only provide the category names, no descriptions needed and each category name is separated by the character \".\""},
        #             ]}]
        #     )
        # text_prompt = completion.choices[0].message.content

        image_pil = Image.fromarray(image)
        raw_image = image_pil.resize((384, 384))
        tagging_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        raw_image = tagging_transform(raw_image).unsqueeze(0).to("cuda:0")
        ram_model.eval()
        text_prompt = inference(raw_image, ram_model)[0].replace(' | ', '.')
        classes = process_tag_classes(text_prompt=text_prompt)

        detections_gd = detection_model.predict_with_classes(    
                image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), # This function expects a BGR image...
                classes=classes,
                box_threshold=0.3,
                text_threshold=0.3
            )
        
        if len(detections_gd.class_id) > 0:
            ### Non-maximum suppression ###
            print(f"Before NMS: {len(detections_gd.xyxy)} boxes")
            nms_idx = ops.nms(
                torch.from_numpy(detections_gd.xyxy), 
                torch.from_numpy(detections_gd.confidence), 
                0.5
            ).numpy().tolist()
            print(f"After NMS: {len(detections_gd.xyxy)} boxes")
            detections_gd.xyxy = detections_gd.xyxy[nms_idx]
            detections_gd.confidence = detections_gd.confidence[nms_idx]
            detections_gd.class_id = detections_gd.class_id[nms_idx]
            
            # Somehow some detections will have class_id=-1, remove them
            # valid_idx = detections.class_id != -1
            valid_idx = [i for i, val in enumerate(detections_gd.class_id) if (val is not None and val != -1)]
            detections_gd.xyxy = detections_gd.xyxy[valid_idx]
            detections_gd.confidence = detections_gd.confidence[valid_idx]
            detections_gd.class_id = detections_gd.class_id[valid_idx]
            
            confidences = detections_gd.confidence
            detection_class_ids = detections_gd.class_id.astype(int)
            xyxy_np = detections_gd.xyxy
            xyxy_tensor = torch.from_numpy(xyxy_np).to("cuda:0")
    else:
        results = detection_model.predict(image, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

    # if there are detections,
    # Get Masks Using SAM or MobileSAM
    # UltraLytics SAM
    if xyxy_tensor.numel() != 0:
        sam_out = sam_predictor.predict(image, bboxes=xyxy_tensor, verbose=False)
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()

    # Create a detections object that we will save later
    curr_det = sv.Detections(
        xyxy=xyxy_np,
        confidence=confidences,
        class_id=detection_class_ids,
        mask=masks_np,
    )
    
    image_crops, image_feats, text_feats = compute_clip_features_batched(
                image, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), device='cuda')
    

    results = {
        # add new uuid for each detection 
        "xyxy": curr_det.xyxy,
        "confidence": curr_det.confidence,
        "class_id": curr_det.class_id,
        "mask": curr_det.mask,
        "classes": obj_classes.get_classes_arr(),
        "image_crops": image_crops,
        "image_feats": image_feats,
        "text_feats": text_feats,
        # "detection_class_labels": detection_class_labels,
        # "labels": labels,
        # "edges": edges,
    }

    resized_gobs = resize_gobs(results, image)

    filtered_gobs = filter_gobs(resized_gobs, image, 
        skip_bg=cfg['skip_bg'],
        BG_CLASSES=obj_classes.get_bg_classes_arr(),
        mask_area_threshold=cfg['mask_area_threshold'],
        max_bbox_area_ratio=cfg['max_bbox_area_ratio'],
        mask_conf_threshold=cfg['mask_conf_threshold'],
    )

    filtered_gobs['mask'] = mask_subtract_contained(filtered_gobs['xyxy'], filtered_gobs['mask'])

    filtered_gobs['mask'] = mask_erosion(filtered_gobs['mask'])

    for idx in range(len(filtered_gobs['mask'])):
        map_object = {
            'idx' : idx + 1,
            'class_id' : [filtered_gobs['class_id'][idx]],
            # 'class_label' : det.class_label,
            'mask' : filtered_gobs['mask'][idx],
            'mask_area' : filtered_gobs['mask'][idx].sum(),
            'clip_ft' : filtered_gobs['image_feats'][idx],
            'image_crops' : filtered_gobs['image_crops'][idx],
            'num_detections' : 1,  # Number of detections for this object
            'best_view' : 0
        }
        objects.append(map_object)

    objects = merge_first_objects_itself(objects, cfg)
    idx_mask = np.zeros((color.shape[0], color.shape[1]), dtype=np.int8) 
    # 构建一个表示特征颜色的tensor
    feature_mask = torch.zeros((color.shape[0], color.shape[1], 3), dtype=torch.float32)

    for i in range(len(objects)):
        mask = objects[i]['mask'] > 0
        idx_mask[mask] = objects[i]['idx']
        feature_mask[mask] = torch.tensor(color_book[objects[i]['idx']], dtype=torch.float32)
        objects[i]['visibility'] = objects[i]['mask_area'] / (color.shape[0] * color.shape[1])
        objects[i]['best_view'] = 0
        # debug   
        # mask_pixel_count = objects[i]['mask'].sum().item()

        # plt.title(f"Boolean Mask:{objects[i]['idx']}")
        # plt.imshow(objects[i]['mask'], cmap='gray')  
        # plt.axis('off')
        # plt.show()

    return idx_mask, feature_mask, objects

def process_tag_classes(text_prompt:str) -> list[str]:
    '''Convert a text prompt from Tag2Text to a list of classes. '''
    classes = text_prompt.split('.')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    add_classes = ["picture","handle", 'Bottled Coke', 'Canned Beer', 'apple', 'potato', 'green toy', 'blue bottle', 'green container', 'blue and grey umbrella', 'blue toy', 'pen', 'orange', 'eggplant', 'yellow bottle', 'corn', 'chili pepper', 'small scissors', 'keys', 'green container', 'cabinet', 'long table', 'big table', 'plate']
    remove_classes = [
        "room", "kitchen", "office", "house", "home", "building", "corner",
        "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
        "apartment", "image", "city", "blue", "skylight", "hallway", 
        "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor",
        "floor", "ladder", "sink", "counter top", "hardwood",
        "shower curtain", "curtain", "slide", "peak", "closet",
        "man", "woman", "child", "boy", "girl", "person", "human", "drawer"]
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    return classes


def process_this_frame_detection(
        image: torch.Tensor, 
        time_idx: int,
        detection_model, 
        ram_model,
        ai_client,
        sam_predictor: SAM, 
        clip_model, 
        clip_preprocess, 
        clip_tokenizer,
        obj_classes,
        cfg
)-> DetectionList:
    detections = DetectionList()
    image = image.cpu().numpy().astype(np.uint8)

    if cfg['detection_model'] == 'groundingdino':
        # img_url = encode_img(image)  # Encode the first image to Base64 format
        # completion = ai_client.chat.completions.create(
        #     model="qwen2.5-vl-72b-instruct",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        #     messages=[{"role": "user","content": [
        #             {"type": "image_url",
        #             "image_url": {"url": f"{img_url}"}},
        #             {"type": "text", "text": "Describe visible object categories in front of you. Only provide the category names, no descriptions needed and each category name is separated by the character \".\""},
        #             ]}]
        #     )
        # text_prompt = completion.choices[0].message.content
        image_pil = Image.fromarray(image)
        raw_image = image_pil.resize((384, 384))
        tagging_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        raw_image = tagging_transform(raw_image).unsqueeze(0).to("cuda:0")
        ram_model.eval()
        text_prompt = inference(raw_image, ram_model)[0].replace(' | ', '.')
        classes = process_tag_classes(text_prompt=text_prompt)

        detections_gd = detection_model.predict_with_classes(    
                image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), # This function expects a BGR image...
                classes=classes,
                box_threshold=0.3,
                text_threshold=0.3
            )
        
        if len(detections_gd.class_id) > 0:
            ### Non-maximum suppression ###
            print(f"Before NMS: {len(detections_gd.xyxy)} boxes")
            nms_idx = ops.nms(
                torch.from_numpy(detections_gd.xyxy), 
                torch.from_numpy(detections_gd.confidence), 
                0.5
            ).numpy().tolist()
            print(f"After NMS: {len(detections_gd.xyxy)} boxes")
            detections_gd.xyxy = detections_gd.xyxy[nms_idx]
            detections_gd.confidence = detections_gd.confidence[nms_idx]
            detections_gd.class_id = detections_gd.class_id[nms_idx]
            
            # Somehow some detections will have class_id=-1, remove them
            # valid_idx = detections.class_id != -1
            valid_idx = [i for i, val in enumerate(detections_gd.class_id) if (val is not None and val != -1)]
            detections_gd.xyxy = detections_gd.xyxy[valid_idx]
            detections_gd.confidence = detections_gd.confidence[valid_idx]
            detections_gd.class_id = detections_gd.class_id[valid_idx]
            
            confidences = detections_gd.confidence
            detection_class_ids = detections_gd.class_id.astype(int)
            xyxy_np = detections_gd.xyxy
            xyxy_tensor = torch.from_numpy(xyxy_np).to("cuda:0")
    else:
        results = detection_model.predict(image, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

    if xyxy_tensor.numel() != 0:
        sam_out = sam_predictor.predict(image, bboxes=xyxy_tensor, verbose=False)
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()
    else:
        return detections


    curr_det = sv.Detections(
        xyxy=xyxy_np,
        confidence=confidences,  # np
        class_id=detection_class_ids,  # np
        mask=masks_np,
    )
    
    image_crops, image_feats, text_feats = compute_clip_features_batched(
                                                image, curr_det, 
                                                clip_model, clip_preprocess, clip_tokenizer, 
                                                obj_classes.get_classes_arr(), device='cuda')
    
    results = {
        # add new uuid for each detection 
        "xyxy": curr_det.xyxy,
        "confidence": curr_det.confidence,
        "class_id": curr_det.class_id,
        "mask": curr_det.mask,
        "classes": obj_classes.get_classes_arr(),
        "image_crops": image_crops,
        "image_feats": image_feats,
        "text_feats": text_feats,
        # "detection_class_labels": detection_class_labels,
        # "labels": labels,
        # "edges": edges,
    }

    resized_gobs = resize_gobs(results, image)

    filtered_gobs = filter_gobs(resized_gobs, image, 
        skip_bg=True,
        BG_CLASSES=obj_classes.get_bg_classes_arr(),
        mask_area_threshold=cfg['mask_area_threshold'],
        max_bbox_area_ratio=cfg['max_bbox_area_ratio'],
        mask_conf_threshold=cfg['mask_conf_threshold'],
    )

    filtered_gobs['mask'] = mask_subtract_contained(filtered_gobs['xyxy'], filtered_gobs['mask'])

    for idx in range(len(filtered_gobs['mask'])):
        mask_area = filtered_gobs['mask'][idx].sum()
        num_labels, _ = cv2.connectedComponents(filtered_gobs['mask'][idx].astype(np.uint8), connectivity=4)
        if mask_area > 500 and (num_labels - 1) < 5 :
            filtered_gobs['mask'][idx] = binary_erosion(filtered_gobs['mask'][idx], structure=np.ones((3, 3)))
            detection = {
                'idx' : idx + 1,
                'class_id' : [filtered_gobs['class_id'][idx]],
                'mask' : filtered_gobs['mask'][idx],
                'mask_area' : filtered_gobs['mask'][idx].sum(),
                'clip_ft' : filtered_gobs['image_feats'][idx],
                'image_crops' : filtered_gobs['image_crops'][idx],
                'num_detections' : 1,  # Number of detections for this object
                'best_view' : time_idx
            }
            detections.append(detection)
    
    if len(detections) > 0:
        detections = merge_curr_detection_itself(detections, cfg)

    return detections


def select_idx_gaussian(params: dict, 
                        select_idx,
                        color_book,
                        if_first_frame):
    
    # TODO：剔除 indice_idx 中 max_probs 小于 0.9 的点

    # 除了 indice 都设为黑色 阻挡问题
    indice = np.where(params['object_idx'] == select_idx)[0]
    
    select_params = {
        'means3D': params['means3D'][indice],
        'rgb_colors': params['rgb_colors'][indice],
        'unnorm_rotations': params['unnorm_rotations'][indice],
        'logit_opacities': params['logit_opacities'][indice],
        'log_scales': params['log_scales'][indice],
        'cam_unnorm_rots': params['cam_unnorm_rots'],
        'cam_trans': params['cam_trans'],
    }

    return select_params


def select_idx_occupy_gaussian(params: dict, 
                        select_idx,
                        color_book,
                        if_first_frame):
    
    # TODO：剔除 indice_idx 中 max_probs 小于 0.9 的点

    # 除了 indice 都设为黑色 阻挡问题
    indice = np.where(params['object_idx'] == select_idx)[0]
    un_indice = np.where(params['object_idx'] != select_idx)[0]
    params['rgb_colors'][indice] = torch.ones_like(params['rgb_colors'][indice])  # Set color to black
    params['rgb_colors'][un_indice] = torch.zeros_like(params['rgb_colors'][un_indice])  # Set color to black

    select_params = {
        'means3D': params['means3D'],
        'rgb_colors': params['rgb_colors'],
        'unnorm_rotations': params['unnorm_rotations'],
        'logit_opacities': params['logit_opacities'],
        'log_scales': params['log_scales'],
        'cam_unnorm_rots': params['cam_unnorm_rots'],
        'cam_trans': params['cam_trans'],
    }

    return select_params


def render_curr_frame_with_idx(params: dict, 
                               time_idx, 
                               curr_data: dict, 
                               objects: MapObjectList,
                               color_book,
                               if_first_frame: bool = False):

    curr_frame_objects = MapObjectList()

    with torch.no_grad():
        for map_object in tqdm(objects):

            select_params = select_idx_gaussian(params, map_object['idx'], color_book, if_first_frame)
            transformed_select_gaussians = transform_to_frame(select_params, time_idx, gaussians_grad=False, camera_grad=False)
            in_frustum = points_in_frustum(transformed_select_gaussians['means3D'], curr_data['intrinsics'], 
                                           curr_data['cam'].image_width, curr_data['cam'].image_height, 0.1, 100)
            select_rendervar = transformed_params_for_detection(select_params, transformed_select_gaussians)
            rasterizer = Renderer(raster_settings=curr_data['cam'])
            object_mask, _, _, = rasterizer(**select_rendervar)
            
            bool_mask = (object_mask != 0).any(dim=0).cpu().numpy()
            mask_pixel_count = bool_mask.sum().item()

            if mask_pixel_count > 200:
                map_object = {
                    'idx' : map_object['idx'],
                    'mask' : bool_mask,
                    'mask_area' : mask_pixel_count,
                    'color_mask' : object_mask,
                    'clip_ft' : map_object['clip_ft'],
                    'num_detections' : map_object['num_detections'],
                    'in_frustum' : in_frustum
                }
                curr_frame_objects.append(map_object)
            else:
                print(f"Skipping object {map_object['idx']} due to low number of pixs({mask_pixel_count}) after splatting in current camera view")
                continue
    
    return curr_frame_objects


def points_in_frustum(points_world, intrinsics, width, height, near, far):
    """
    判断多个点是否在相机视锥体内
    
    参数:
    points_world : np.array (N, 3) - 世界坐标系中的多个3D点
    width : int - 图像宽度(像素)
    height : int - 图像高度(像素)
    near : float - 近裁剪面距离
    far : float - 远裁剪面距离
    
    返回:
    in_frustum : np.array (N,) bool - 每个点是否在视锥体内
    """
    # 转换为numpy数组
    points_cam = points_world
    
    # 提取坐标分量
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    
    # 深度检查 (向量化)
    depth_ok = (z > 0) & (z >= near) & (z <= far)
    
    # 提取内参
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 计算投影坐标 (避免除零错误)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
    
    # 图像边界检查 (向量化)
    in_image = (u >= 0) & (u <= width) & (v >= 0) & (v <= height)
    
    # 综合条件：深度有效且在图像范围内
    in_frustum = depth_ok & in_image
    
    # 处理无效点（z<=0）
    in_frustum[torch.isnan(u) | torch.isnan(v)] = False

    return in_frustum


def update_curr_object_visibility(params: dict, 
                                  time_idx, 
                                  curr_data: dict, 
                                  objects: MapObjectList,
                                  color_book,
                                  if_first_frame: bool = False):

    with torch.no_grad():
        for i, map_object in enumerate(objects):

            params_wb = select_idx_occupy_gaussian(params, map_object['idx'], color_book, if_first_frame)
            transformed_wb_gaussians = transform_to_frame(params_wb, time_idx, gaussians_grad=False, camera_grad=False)
            
            select_params = select_idx_gaussian(params, map_object['idx'], color_book, if_first_frame)
            transformed_select_gaussians = transform_to_frame(select_params, time_idx, gaussians_grad=False, camera_grad=False)
            
            rasterizer = Renderer(raster_settings=curr_data['cam'])
            # visible = rasterizer.markVisible(transformed_select_gaussians['means3D'])
            visible = points_in_frustum(transformed_select_gaussians['means3D'], curr_data['intrinsics'], 
                                           curr_data['cam'].image_width, curr_data['cam'].image_height, 0.1, 100)

            select_rendervar = transformed_params_for_detection(params_wb, transformed_wb_gaussians)
            object_mask, _, _, = rasterizer(**select_rendervar)
            bool_mask = (object_mask != 0).any(dim=0).cpu().numpy()
            mask_pixel_count = bool_mask.sum().item()

            visibility = (mask_pixel_count / (curr_data['cam'].image_width * curr_data['cam'].image_height)) * (torch.sum(visible) / transformed_select_gaussians['means3D'].shape[0])
                
            if mask_pixel_count > 200:
                if "visibility" not in objects[i]:
                    objects[i]['visibility'] = visibility
                    objects[i]['best_view'] = time_idx
                else:
                    if visibility > map_object['visibility']:
                        objects[i]['visibility'] = visibility 
                        objects[i]['best_view'] = time_idx
                        objects[i]['mask'] = bool_mask
                        objects[i]['mask_area'] = mask_pixel_count
                        # plt.title("Boolean Mask")
                        # plt.title(f"Bool mask : {map_object['idx']}")  
                        # plt.imshow(bool_mask, cmap='gray')  
                        # plt.axis('off')
                        # plt.show()

    
    return objects

def update_curr_object_visibility_1(params: dict, 
                                  time_idx, 
                                  curr_data: dict, 
                                  objects: MapObjectList,
                                  color_book,
                                  if_first_frame: bool = False):

    with torch.no_grad():
        for i, map_object in enumerate(objects):

            select_params = select_idx_occupy_gaussian(params, map_object['idx'], color_book, if_first_frame)

            transformed_select_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
            select_rendervar = transformed_params_for_detection(params, transformed_select_gaussians)
            rasterizer = Renderer(raster_settings=curr_data['cam'])
            object_mask, _, _, = rasterizer(**select_rendervar)
            visible = rasterizer.markVisible(select_params['means3D'])
            
            bool_mask = (object_mask != 0).any(dim=0).cpu().numpy()
            mask_pixel_count = bool_mask.sum().item()

            visibility = (mask_pixel_count / (curr_data['cam'].image_width * curr_data['cam'].image_height)) * (torch.sum(visible) / transformed_select_gaussians['means3D'].shape[0])
                
            if mask_pixel_count > 200:
                if "visibility" not in objects[i]:
                    objects[i]['visibility'] = visibility
                    objects[i]['best_view'] = time_idx
                else:
                    if visibility > map_object['visibility']:
                        objects[i]['visibility'] = visibility 
                        objects[i]['best_view'] = time_idx
                        objects[i]['mask'] = bool_mask
                        objects[i]['mask_area'] = mask_pixel_count
                    plt.title("Boolean Mask")
                    plt.title(f"Bool mask : {map_object['idx']}")  
                    plt.imshow(bool_mask, cmap='gray')  
                    plt.axis('off')
                    plt.show()

    
    return objects


def compute_mask_iou_similarities(detections: DetectionList, 
                                  curr_cam_mapobjects: MapObjectList,
                                  privilege: bool = False):

    detection_masks = detections.get_stacked_values_torch('mask').unsqueeze(1)
    mapobject_masks = curr_cam_mapobjects.get_stacked_values_torch('mask').unsqueeze(0).cpu()

    intersection = (detection_masks & mapobject_masks).float().sum(dim=(2, 3))
    union = (detection_masks | mapobject_masks).float().sum(dim=(2, 3))
    iou = intersection / union

    if privilege:
        # 计算mapobject占detection的比例：地图物体被detection完全覆盖
        map_sum = mapobject_masks.float().sum(dim=(2, 3))
        object_in_detection = intersection / map_sum
        return iou, object_in_detection, map_sum
    
    return iou


def compute_clip_features_similarities(detections: DetectionList, 
                                       curr_cam_mapobjects: MapObjectList):
    
    detection_features = detections.get_stacked_values_torch('clip_ft').unsqueeze(-1)
    mapobject_features = curr_cam_mapobjects.get_stacked_values_torch('clip_ft').T.unsqueeze(0)

    clip_features_similarities = F.cosine_similarity(detection_features, mapobject_features, dim=1)

    return clip_features_similarities
    

def aggregate_similarities(iou_similarities: torch.Tensor, 
                           features_similarities: torch.Tensor,
                           similarity_bias: float):
    
    similarities = (1 + similarity_bias) * iou_similarities + (1 - similarity_bias) * features_similarities

    return similarities / 2 


def match_detections_to_mapobjects(similarities: torch.Tensor,
                                   curr_cam_mapobjects: MapObjectList,
                                   objects: MapObjectList,
                                   similarity_threshold: float):
    
    match_indices = []
    for detected_obj_idx in range(similarities.shape[0]):
        max_sim_value = similarities[detected_obj_idx].max()
        if max_sim_value <= similarity_threshold:
            match_indices.append(None)
        else:
            i = curr_cam_mapobjects[similarities[detected_obj_idx].argmax().item()]['idx']
            for index, object in enumerate(objects):
                if object['idx'] == i:
                    match_indices.append(index)
                    break

    return match_indices


def merge_obj2_into_obj1(obj1: dict, obj2: dict):

    obj1['class_id'].extend(obj2['class_id'])

    merged_obj = {
        'idx' : obj1['idx'],
        'class_id' : obj1['class_id'],
        'mask' : obj1['mask'] | obj2['mask'],
        'mask_area' : (obj1['mask'] | obj2['mask']).sum(),
        'clip_ft' : (obj1['clip_ft'] * obj1['num_detections'] + obj2['clip_ft'] * obj2['num_detections']) / (obj1['num_detections'] + obj2['num_detections']),
        'image_crops' : obj1['image_crops'],
        'num_detections' : obj1['num_detections'] + obj2['num_detections'],
        'best_view' : obj1['best_view']
    }

    return merged_obj

def merge_obj2_into_obj1_for_merge(obj1: dict, obj2: dict):

    if obj1['mask_area'] > obj2['mask_area']:
        mask_to_keep = obj1['mask']
        mask_area_to_keep = obj1['mask_area']
        best_view_to_keep = obj1['best_view']
        image_crops_to_keep = obj1['image_crops']
    else:
        mask_to_keep = obj2['mask']
        mask_area_to_keep = obj2['mask_area']
        best_view_to_keep = obj2['best_view']
        image_crops_to_keep = obj2['image_crops']

    obj1['class_id'].extend(obj2['class_id'])
    merged_obj = {
        'idx' : obj1['idx'],
        'class_id' : obj1['class_id'],
        'mask' : mask_to_keep,
        'mask_area' : mask_area_to_keep,
        'clip_ft' : (obj1['clip_ft'] * obj1['num_detections'] + obj2['clip_ft'] * obj2['num_detections']) / (obj1['num_detections'] + obj2['num_detections']),
        'image_crops' : image_crops_to_keep,
        'num_detections' : obj1['num_detections'] + obj2['num_detections'],
        'best_view' : best_view_to_keep
        # 'visibility' : obj1['visibility'],
        # 'best_view' : obj1['best_view']
    }

    return merged_obj


def merge_obj_matches(detections: DetectionList, 
                      objects: MapObjectList,
                      curr_cam_mapobjects: MapObjectList,  
                      match_indices: list,
                      privilege_similarities: torch.Tensor,
                      currmap_sum: torch.Tensor,
                      time_idx):
    new_objects_idx = []
    privilege_object = []
    for detected_obj_idx, existing_obj_match_idx in enumerate(match_indices):
        if existing_obj_match_idx is None:
            if privilege_similarities[detected_obj_idx].max() > 0.7:
                # 虽然没匹配上地图中的物体，但是覆盖了地图中的某个物体，将该物体对应到地图物体的idx
                idx = curr_cam_mapobjects[privilege_similarities[detected_obj_idx].argmax().item()]['idx']
                for index, object in enumerate(objects):
                    if object['idx'] == idx:
                        merged_obj = merge_obj2_into_obj1_for_merge(
                            obj1=objects[index],
                            obj2=detections[detected_obj_idx],
                        )
                        objects[index] = merged_obj
                        match_indices[detected_obj_idx] = index
                        # with open('/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/new_objects_idx.txt', 'a') as f:
                        #     f.write(f" 覆盖了：{idx}\n")  
                        break
            else:
                # track the new object detection:TODO:增加一个记录当前最大idx的变量
                detections[detected_obj_idx]['idx'] = objects[-1]['idx'] + 1
                objects.append(detections[detected_obj_idx])
                match_indices[detected_obj_idx] = len(objects) - 1
                new_objects_idx.append(detections[detected_obj_idx]['idx'])
        else:
            detected_obj = detections[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            merged_obj = merge_obj2_into_obj1_for_merge(
                obj1=matched_obj,
                obj2=detected_obj,
            )
            objects[existing_obj_match_idx] = merged_obj

            # 如果当前匹配到了地图中的物体，但是覆盖了地图中的其他物体
            # 赋予被匹配到的地图物体特权
            if privilege_similarities[detected_obj_idx].max() > 0.98 and currmap_sum[0][privilege_similarities[detected_obj_idx].argmax().item()] > 1000:
            # if privilege_similarities[detected_obj_idx].max() > 0.98 :
                idx = curr_cam_mapobjects[privilege_similarities[detected_obj_idx].argmax().item()]['idx']
                for index, object in enumerate(objects):
                    if object['idx'] == idx and (index != existing_obj_match_idx):
                        # tuple: (privilege_obj_idx, new_obj_idx) clip_ft
                        clip_sim = F.cosine_similarity(torch.from_numpy(objects[existing_obj_match_idx]['clip_ft']), 
                                                       torch.from_numpy(objects[index]['clip_ft']), dim=0)
                        if clip_sim > 0.3:
                            privilege_object.append((objects[existing_obj_match_idx]['idx'], idx))
                            
                            img_crop = np.array(detections[detected_obj_idx]['image_crops'])
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(f"/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/{time_idx}_{detected_obj_idx}.jpg", img_crop)

                            # 将mask保存为图片
                            mask = curr_cam_mapobjects[privilege_similarities[detected_obj_idx].argmax().item()]['mask']  # np (680, 1200) bool
                            mask = mask.astype(np.uint8) * 255
                            # cv2.imwrite(f"/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/{time_idx}_{privilege_similarities[detected_obj_idx].argmax().item()}_mask.jpg", mask)
                            img_crop = np.array(objects[existing_obj_match_idx]['image_crops'])
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(f"/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/{privilege_object[0][0]}.jpg", img_crop)
                            img_crop = np.array(objects[index]['image_crops'])
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(f"/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/{privilege_object[0][1]}.jpg", img_crop)

                            # with open('/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/new_objects_idx.txt', 'a') as f:
                            #     f.write(f"合并了：{privilege_object} {clip_sim}\n")   
                        break
                

    return objects, match_indices, new_objects_idx, privilege_object


def compute_similarities_and_merge(detections: DetectionList, 
                                   curr_cam_mapobjects: MapObjectList,
                                   objects: MapObjectList,
                                   color_book,
                                   cfg: dict,
                                   time_idx,
                                   H, W):
    # TODO: 当前帧检测到mask太小的目标，不参与合并，避免影响匹配！！！
    # 计算detections与当前帧的mapobjects之间的相似度
    '''
        TODO: 是否增加3d重合度的计算？ 后期再考虑
              添加计算mapobject占detection的比例：
                    地图物体被detection完全覆盖，并且有一定相似度，将地图中的物体合并到detection中
                    privilege detection 允许更新被占用的点
    '''
    if len(detections) == 0:
        curr_objects_idx = []
        new_objects_idx = []
        privilege_object = []
        curr_idx_mask = np.zeros((H, W), dtype=np.int8) 
        curr_features_mask = torch.zeros((H, W, 3), dtype=torch.float32)

        return curr_idx_mask, curr_features_mask, objects, curr_objects_idx, new_objects_idx, privilege_object


    if len(curr_cam_mapobjects) != 0 :
        iou_similarities, privilege_similarities, currmap_sum = compute_mask_iou_similarities(detections, curr_cam_mapobjects, True)
        features_similarities = compute_clip_features_similarities(detections, curr_cam_mapobjects)
        similarities = aggregate_similarities(iou_similarities, features_similarities, cfg['similarity_bias'])
        # 得到detecions与objects的匹配关系
        match_indices = match_detections_to_mapobjects(similarities, curr_cam_mapobjects, objects, cfg['similarity_threshold'])
        # 将当前帧合并到地图中
        objects, match_indices, new_objects_idx, privilege_object = merge_obj_matches(detections, objects, curr_cam_mapobjects, 
                                                                                      match_indices, privilege_similarities, currmap_sum, time_idx)
    else:
        match_indices = []
        new_objects_idx = []
        privilege_object = []
        for detected_obj_idx, _ in enumerate(detections):
            detections[detected_obj_idx]['idx'] = objects[-1]['idx'] + 1
            objects.append(detections[detected_obj_idx])
            match_indices.append(len(objects) - 1)
            new_objects_idx.append(detections[detected_obj_idx]['idx'])


    curr_idx_mask = np.zeros((detections[0]['mask'].shape[0], detections[0]['mask'].shape[1]), dtype=np.int8) 
    curr_features_mask = torch.zeros((detections[0]['mask'].shape[0], detections[0]['mask'].shape[1], 3), dtype=torch.float32)
    curr_objects_idx = []
    for detected_obj_idx, existing_obj_match_idx in enumerate(match_indices):
        mask = detections[detected_obj_idx]['mask'] > 0
        curr_idx_mask[mask] = objects[existing_obj_match_idx]['idx']
        curr_features_mask[mask] = torch.tensor(color_book[objects[existing_obj_match_idx]['idx']], dtype=torch.float32)
        curr_objects_idx.append(objects[existing_obj_match_idx]['idx'])
        # # debug
        # plt.subplot(1, 2, 1) 
        # plt.title("Boolean Mask")
        # plt.imshow(detections[detected_obj_idx]['mask'], cmap='gray')  
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.title(f"curr_cam_mapobjects Mask")
        # for obj in curr_cam_mapobjects:
        #     if obj['idx'] == objects[existing_obj_match_idx]['idx']:
        #         plt.imshow(obj['mask'], cmap='gray')  
        #         break
        # plt.axis('off')
        
        # plt.tight_layout()  # 自动调整子图间距
        # plt.show()

    return curr_idx_mask, curr_features_mask, objects, curr_objects_idx, new_objects_idx, privilege_object


def check_update(color, depth, curr_cam_mapobjects, detections=None, cfg=None):

    objects_to_remove = []
    for i, curr_obj in enumerate(curr_cam_mapobjects):
        mask = torch.from_numpy(curr_obj['mask']).cuda()
        weighted_im = curr_obj['color_mask'][:,mask].cpu().numpy()  
        weighted_gt_im = color[:,mask].cpu().numpy()
        
        ssim_scores = 0
        for c in range(3):  # 遍历R、G、B通道
            gt_channel_masked = weighted_gt_im[c, ...]
            pred_channel_masked = weighted_im[c, ...] 
            # 计算SSIM（需将数据范围归一化到0-1或0-255）
            score = ssim(
                gt_channel_masked, pred_channel_masked,
                data_range=1,  # 如果图像是uint8类型，范围为0-255
                channel_axis=None  # 单通道无需指定
            )
            ssim_scores += score

        ssim_scores = ssim_scores / 3
        if ssim_scores < 0.15:
            objects_to_remove.append(curr_obj['idx'])

        vis1 =  curr_obj['color_mask'] * mask
        vis2 = color * mask
        plt.subplot(1, 2, 1) 
        plt.title("dynamic changes")
        plt.imshow(vis1.permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')

        plt.subplot(1, 2, 2) 
        plt.title("dynamic changes gt")
        plt.imshow(vis2.permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')
        plt.tight_layout()  # 自动调整子图间距
        plt.show()

    # iou_similarities = compute_mask_iou_similarities(curr_cam_mapobjects, detections, False)
    # features_similarities = compute_clip_features_similarities(curr_cam_mapobjects, detections)
    # similarities = aggregate_similarities(iou_similarities, features_similarities, cfg['similarity_bias'])
    # objects_to_remove = []
    # for i, curr_obj in enumerate(curr_cam_mapobjects):
    #     if similarities[i].max() < cfg['similarity_threshold']:
    #         objects_to_remove.append(curr_obj['idx'])
    
    return objects_to_remove


def get_curr_objects_pcd(depth, idx_mask, curr_objects_idx, intrinsics, w2c, transform_pts=True, mask=None):

    width, height = depth.shape[2], depth.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
        # pts = pts_cam @ w2c[:3, :3].T + w2c[:3, 3]
    else:
        pts = pts_cam

    point_cld = torch.cat((pts, idx_mask.reshape(-1, 1).to('cuda')), -1)

    if mask is not None:
        point_cld = point_cld[mask]

    curr_objects_pcd = []
    for idx in curr_objects_idx:
        mask = point_cld[:, 3] == idx
        pcd = {
            'idx': idx,
            'points': point_cld[mask][:, :3].cpu().numpy(),
        }   
        curr_objects_pcd.append(pcd)

    return curr_objects_pcd


def update_curr_objects_gaussians(params : dict, objects: MapObjectList,
                                  curr_objects_pcd, new_objects_idx, privilege_object_tuple, cfg, time_idx):

    scene_pcd_np = params['means3D'].detach().cpu().numpy()

    index = faiss.IndexFlatL2(3)
    index.add(scene_pcd_np)

    privilege_object = [obj[0] for obj in privilege_object_tuple]

    invaild_new_objects_idx = []
    for i in range(len(curr_objects_pcd)):
        object_idx = curr_objects_pcd[i]['idx']
        object_pcd_np = curr_objects_pcd[i]['points']

        _, indices = index.search(object_pcd_np, 1)

        indices = indices.flatten()
        
        print(f"update gs num before: {indices.shape[0]}")

        update_positions = ((params['object_idx'][indices] == 0) | (params['object_idx'][indices] == curr_objects_pcd[i]['idx']))

        if object_idx in new_objects_idx:
            if (indices.shape[0] > cfg['update_gs_num_threshold']) and ((np.sum(update_positions) / indices.shape[0]) > cfg['update_gs_ratio_threshold']):

                params['object_idx'][indices[update_positions.flatten()]] = object_idx
                print(f"update new object {object_idx} gs num:{indices.shape[0]} {np.sum(update_positions)}  {(np.sum(update_positions) / indices.shape[0])}")
            else:
                invaild_new_objects_idx.append(object_idx)
                print(f"invaild_new_object:{indices.shape[0]} {np.sum(update_positions) / indices.shape[0]}")
        else:
            if object_idx in privilege_object:
                for obj in privilege_object_tuple:
                    if obj[0] == object_idx:
                        associated_object_idx = obj[1]
                        break
                associated_indices = np.where(params['object_idx'] == associated_object_idx)[0]
                params['object_idx'][indices[update_positions.flatten()]] = object_idx
                params['object_idx'][associated_indices] = object_idx
                print(f"updated privilege object {object_idx} gs num: {np.sum(update_positions)}")
            else:
                params['object_idx'][indices[update_positions.flatten()]] = object_idx
                print(f"updated gs num: {np.sum(update_positions)}")

    return params, invaild_new_objects_idx


def slice_invaild_new_objects(invaild_new_objects_idx, objects: MapObjectList, curr_objects_idx, curr_data: dict):
    
    if len(invaild_new_objects_idx) == 0:
        return objects, curr_objects_idx, curr_data
    else:
        for invaild_idx in invaild_new_objects_idx:
            invaild_pix =  curr_data['idx_mask'] == invaild_idx
            curr_data['idx_mask'][invaild_pix] = 0
            curr_data['features'][invaild_pix] = 0

        objects = [obj for obj in objects if obj['idx'] not in invaild_new_objects_idx]
        curr_obj_idx = [cidx for cidx in curr_objects_idx if cidx not in invaild_new_objects_idx]
        
    return  MapObjectList(objects), curr_obj_idx, curr_data


def update_objects_idx(params, objects: MapObjectList, curr_objects_idx, all=False):

    if all:
        for obj in objects:
            obj['all_idx'] = np.where(params['object_idx'] == obj['idx'])[0]
    else:
        for obj in objects:
            if obj['idx'] not in curr_objects_idx:
                continue
            else:
                obj['all_idx'] = np.where(params['object_idx'] == obj['idx'])[0]    

    return objects


def save_keyframe_list(keyframe_list: list, output_dir):
    
    keyframe_list_save_path = Path(output_dir) / "keyframelist.pkl.gz"
    keyframe_list_save_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(keyframe_list_save_path, "wb") as f:
        pickle.dump(keyframe_list, f)

    print(f"Saving keyframe list to: {keyframe_list_save_path}")


def save_objects(params, objects: MapObjectList, dataset, ai_client, lf_config, output_dir):

    prompt_modes = {
            "focal_prompt": "full+focal_crop",
        }
    dam_model = DescribeAnythingModel(
        model_path=lf_config['dam_model_path'],
        conv_mode=lf_config['dam_conv_mode'],
        prompt_mode=prompt_modes.get(lf_config['dam_prompt_mode'], lf_config['dam_prompt_mode']),
    ).to("cuda:0")

    with open(lf_config['obj_prompt_file'], "r") as f:
        obj_category_prompt = f.read()
        
    with open(lf_config['obj_caption_file'], "r") as f:
        obj_caption_prompt = f.read()
        
    objects_img_crop_save_path = f"{output_dir}/objects_img_crop"

    if not os.path.exists(objects_img_crop_save_path):
        os.makedirs(objects_img_crop_save_path)

    # image_crops PIL to numpy 
    for i, obj in enumerate(objects):
        img_crop = np.array(obj['image_crops'])
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{objects_img_crop_save_path}/{obj['idx']}.jpg", img_crop)

        image, _ ,_ ,_ = dataset[obj['best_view']]
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., obj['best_view']].detach())
        curr_cam_tran = params['cam_trans'][..., obj['best_view']].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran

        outputs = dam_model.get_description(
                Image.fromarray(image.cpu().numpy().astype(np.uint8)),
                obj['mask'],
                "<image>\nDescribe the masked region in detail.",
                temperature=0.2,
                top_p=0.5,
                num_beams=1,
                max_new_tokens=512,
            )
        objects[i]['description'] = outputs
        objects[i]['best_view_w2c'] = curr_w2c

        obj_category_parse_prompt = f"Query description: {outputs}"
        messages = [
            {"role": "system", "content": obj_category_prompt},
            {"role": "user", "content": [{"type": "text", "text": obj_category_parse_prompt}]},
        ]
        chat_response = ai_client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
        answer = chat_response.choices[0].message.content
        answer = answer.replace("'", '"')
        answer = json.loads(answer)
        obj['category'] = answer['Object category']

        obj_caption_parse_prompt = f"Query caption: {obj['category']}"
        messages = [
            {"role": "system", "content": obj_caption_prompt},
            {"role": "user", "content": [{"type": "text", "text": obj_caption_parse_prompt}]},
        ]
        chat_response = ai_client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
        answer = chat_response.choices[0].message.content
        answer = answer.replace("'", '"')
        answer = json.loads(answer)
        obj['caption'] = answer['Object caption']
    
    objects = objects.to_serializable()
    objects_save_path = Path(output_dir) / "objects.pkl.gz"
    objects_save_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(objects_save_path, "wb") as f:
        pickle.dump(objects, f)

    print(f"Saving map objects to: {objects_save_path}")


def load_data(file_path: Path) -> list:
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {file_path}")
    return data


def denoise_and_filter_objects(params, objects, curr_objects_idx, cfg):
    to_remove = []
    for obj in objects:
        indices = np.where(params['object_idx'] == obj['idx'])[0]

        if indices.shape[0] < 100:
            to_remove.append(obj['idx'])
            continue
    
    objects = [obj for obj in objects if obj['idx'] not in to_remove]
    objects = MapObjectList(objects)


    for curr_object_idx in curr_objects_idx:
        indices = torch.from_numpy(np.where(params['object_idx'] == curr_object_idx)[0]).to('cuda')
        object_pcd_np = params['means3D'][indices].detach().cpu().numpy()
        object_pcd_col_np = params['rgb_colors'][indices].detach().cpu().numpy()
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
        object_pcd.colors = o3d.utility.Vector3dVector(object_pcd_col_np)
        cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
        
        # 获取离群点的索引
        outlier_indices = np.setdiff1d(np.arange(len(object_pcd.points)), np.array(ind))
        
        # 转换离群点索引为原始索引
        outlier_indices_original = indices[outlier_indices].detach().cpu()
        
        # 将离群点的 object_idx 设置为 0
        params['object_idx'][outlier_indices_original] = 0


    # # denoise
    # for curr_object_idx in curr_objects_idx:
    #     indices = torch.from_numpy(np.where(params['object_idx'] == curr_object_idx)[0]).to('cuda')
    #     object_pcd_np = params['means3D'][indices].detach().cpu().numpy()
    #     object_pcd_col_np = params['rgb_colors'][indices].detach().cpu().numpy()
    #     object_pcd = o3d.geometry.PointCloud()
    #     object_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
    #     object_pcd.colors = o3d.utility.Vector3dVector(object_pcd_col_np)

    #     object_pcd_clusters = object_pcd.cluster_dbscan(eps=1.0, min_points=10,)
    #     object_pcd_clusters = np.array(object_pcd_clusters)
    #     counter = Counter(object_pcd_clusters)

    #     # Remove the noise label
    #     if counter and (-1 in counter):
    #         del counter[-1]

    #     if counter:
    #         # Find the label of the largest cluster
    #         most_common_label, _ = counter.most_common(1)[0]
            
    #         # Create mask for points in the largest cluster
    #         invaild_mask = object_pcd_clusters != most_common_label

    #         # Apply mask
    #         outlier_indices_original = indices[invaild_mask].detach().cpu().numpy()
    #         params['object_idx'][outlier_indices_original] = 0         



    
    return params


def regular_objects(objects: MapObjectList, cfg):
    pass