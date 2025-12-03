from PIL import Image
import numpy as np
import supervision as sv
import scipy.ndimage as ndi
from openai import OpenAI
from utils.slam_classes import MapObjectList
import cv2
import os
import re
import ast
import base64

# Function to encode the image as base64
def encode_image_for_openai(image_path: str, resize = False, target_size: int=512):
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not resize:
        # Open the image
        print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        return encoded_image
    
    print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        # Determine scaling factor to maintain aspect ratio
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width} x {original_height}")
        
        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        print(f"Resized image dimensions: {new_width} x {new_height}")

        # Resizing the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print("Image resized successfully.")
        
        # Convert the image to bytes and encode it in base64
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            print("Resized image saved temporarily for encoding.")
        
        # Open the temporarily saved image for base64 encoding
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        
        # Clean up the temporary file
        os.remove("temp_resized_image.jpg")
        print("Temporary file removed.")

    return encoded_image

def vlm_extract_object_captions(text: str):
    # Replace newlines with spaces for uniformity
    text = text.replace('\n', ' ')
    
    # Pattern to match the list of objects
    pattern = r'\[(.*?)\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Try to convert the entire string to a list of dictionaries
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If the whole string conversion fails, process each element individually
            elements = re.findall(r'{.*?}', list_str)
            result = []
            for element in elements:
                try:
                    obj = ast.literal_eval(element)
                    if isinstance(obj, dict):
                        result.append(obj)
                except (ValueError, SyntaxError):
                    print(f"Error processing element: {element}")
            return result
    else:
        # No matching pattern found
        print("No list of objects found in the text.")
        return []

def annotate_for_vlm(
    color_path: str, 
    detections: sv.Detections,
    obj_classes, 
    labels: list[str], 
    save_path=None, 
    color: tuple=(0, 255, 0), 
    thickness: int=2, 
    text_color: tuple=(255, 255, 255), 
    text_scale: float=0.6, 
    text_thickness: int=2, 
    text_bg_color: tuple=(255, 255, 255), 
    text_bg_opacity: float=0.95,  # Opacity from 0 (transparent) to 1 (opaque)
    small_mask_threshold = 0.002,
    mask_opacity: float = 0.2  # Opacity for mask fill
) -> np.ndarray:
    annotated_image = cv2.imread(color_path) # This will in BGR color space
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # if image.shape[0] > 700:
    #     print(f"Line 604, image.shape[0]: {image.shape[0]}")
    #     text_scale = 2.5
    #     text_thickness = 5
    total_pixels = annotated_image.shape[0] * annotated_image.shape[1]
    small_mask_size = total_pixels * small_mask_threshold
    
    detections_mask = [detection['mask'] for detection in detections]
    # detections_mask = detections['mask']

    # Sort detections by mask area, large to small, and keep track of original indices
    mask_areas = [np.count_nonzero(mask) for mask in detections_mask]
    sorted_indices = sorted(range(len(mask_areas)), key=lambda x: mask_areas[x], reverse=True)
    
    # Iterate over each mask and corresponding label in the detections in sorted order
    for i in sorted_indices:
        mask = detections_mask[i]
        label = labels[i]
        label_num = label.split(" ")[-1]
        label_name = re.sub(r'\s*\d+$', '', label).strip()
        # bbox = detections[i]['xyxy']
        # bbox = detections['xyxy'][i]

        # draw bounding box
        # cv2.rectangle(annotated_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]),), color, thickness)
        
        obj_color = obj_classes.get_class_color(int(detections[i]['class_id'][0]))
        # obj_color = obj_classes.get_class_color(int(detections['class_id'][i]))
        # multiply by 255 to convert to BGR
        obj_color = tuple([int(c * 255) for c in obj_color])
        
        # Add color over mask for this object 
        mask_uint8 = mask.astype(np.uint8)
        mask_color_image = np.zeros_like(annotated_image)
        mask_color_image[mask_uint8 > 0] = obj_color
        # cv2.addWeighted(annotated_image, 1, mask_color_image, mask_opacity, 0, annotated_image)

        # Draw contours
        contours, _ = cv2.findContours(mask_uint8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_image, contours, -1, obj_color, thickness)

        # # Determine if the mask is considered "small"
        # if mask_areas[i] < small_mask_size:
        #     x_center = int(bbox[2])  # Place the text to the right of the bounding box
        #     y_center = int(bbox[1])  # Place the text above the top of the bounding box
        # else:
        #     # Calculate the centroid of the mask
        #     ys, xs = np.nonzero(mask)
        #     y_center, x_center = ndi.center_of_mass(mask)
        #     x_center, y_center = int(x_center), int(y_center)

        # Determine if the mask is considered "small"
        
        # Calculate the centroid of the mask
        y_center, x_center = ndi.center_of_mass(mask)
        x_center, y_center = int(x_center), int(y_center)
        if mask_areas[i] < small_mask_size:
            # Move y_center upward by a fixed amount for small masks
            y_center = y_center - 20  # Move text 20 pixels above the centroid

        # Prepare text background
        text = label_num + ": " + label_name 
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        text_x_left = x_center - text_width // 2
        text_y_top = y_center + (text_height) // 2
        
        # Create a rectangle sub-image for the text background
        b_pad = 2 # background rectangle padding
        rect_top_left = (text_x_left - b_pad, text_y_top - text_height - baseline - b_pad)
        rect_bottom_right = (text_x_left + text_width + b_pad, text_y_top - baseline//2 + b_pad)
        sub_img = annotated_image[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
        
        # Create the background rectangle with the specified color and opacity
        # make the text bg color be the negative of the text color
        text_bg_color = tuple([255 - c for c in obj_color])
        # now make text bg color grayscale
        text_bg_color = tuple([int(sum(text_bg_color) / 3)] * 3)
        background_rect = np.full(sub_img.shape, text_bg_color, dtype=np.uint8)
        # cv2.addWeighted(sub_img, 1 - text_bg_opacity, background_rect, text_bg_opacity, 0, sub_img)

        # Draw text with background
        cv2.putText(
            annotated_image, 
            text, 
            (text_x_left, text_y_top - baseline), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale, 
            # obj_color,
            # (255,255,255),
            (0,0,0),
            text_thickness, 
            cv2.LINE_AA
        )
        
        # Draw text with background
        cv2.putText(
            annotated_image, 
            text, 
            (text_x_left, text_y_top - baseline), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale,
            # (0,0,0), 
            obj_color,
            text_thickness - 1, 
            cv2.LINE_AA
        )
        
    if save_path:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, rgb_image)

    return annotated_image, sorted_indices

def get_obj_captions_from_image_gpt4v(image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global system_prompt
    
    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."
    
    messages=[
        {
            "role": "system",
            "content": system_prompt_captions
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    vlm_answer_captions = []
    try:
        response = openai_client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages
        )
        
        vlm_answer_str = response.choices[0].message.content
        print(f"vlm_answer_str: {vlm_answer_str}")
        
        vlm_answer_captions = vlm_extract_object_captions(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    
    return vlm_answer_captions

def consolidate_captions(objects: MapObjectList):
    # Formatting the captions into a single string prompt
    captions_text = []
    for obj in objects:
        captions = obj['captions'].copy()
        if 'descriptive' in captions:
            del captions['descriptive']
        captions_text.append(captions)

    user_query = f"{captions_text}"

    messages = [
        {
            "role": "system",
            "content": system_prompt_consolidate_captions
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    consolidated_caption = ""
    try:
        response = openai_client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            # response_format={"type": "json_object"}
        )

        consolidated_caption = response.choices[0].message.content.strip()
        print(f"Consolidated Caption: {consolidated_caption}")

        vlm_consolidated_answer_captions = vlm_extract_object_captions(consolidated_caption)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        vlm_consolidated_answer_captions = ""

    return vlm_consolidated_answer_captions

def query_with_llm(objects: MapObjectList, query: str):
    # Formatting the captions into a single string prompt
    captions_text = [{'idx': obj['idx'], 'name': obj['category'], 'support': obj['caption']} for obj in objects]

    system_prompt = query_with_llm_prompts
    user_query = f"This is the query that you need to answer: {query}\nAnd this is the list of objects in the room: \n{captions_text}"

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    answer = ""
    try:
        response = openai_client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            # response_format={"type": "json_object"}
        )

        answer = response.choices[0].message.content.strip()
        print(f"Candidate objects are :\n{answer}")

        vlm_answer = vlm_extract_object_captions(answer)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        vlm_answer = ""

    return vlm_answer

def descriptive_query_with_llm(objects: MapObjectList, query: str, scene_graph: dict = None, flag: bool = False):
    # Formatting the captions into a single string prompt
    # captions_text = "\n".join([f"{obj['captions']}" for obj in objects])
    captions_text = [{'idx': obj['idx'], 'name': obj['category'], 'support': obj['caption']} for obj in objects]

    user_query = f"This is the query that you need to answer: {query}\nAnd this is the list of objects in the room: \n{captions_text}"
    user_query_with_scene_graph = f"This is the query that you need to answer: {query}\nAnd this is the list of objects in the room: \n{captions_text}\nIn addition, this is the hierarchical relationships of the objects: \n{scene_graph}"

    system_prompt = descriptive_query_with_llm_and_scene_graph_prompts if flag else descriptive_query_with_llm_prompts
    user_query = user_query_with_scene_graph if flag else user_query

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    answer = ""
    try:
        response = openai_client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            # response_format={"type": "json_object"}
        )

        answer = response.choices[0].message.content.strip()
        print(f"Candidate objects are :\n{answer}")

        vlm_answer = vlm_extract_object_captions(answer)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        vlm_answer = ""

    return vlm_answer

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

gpt_model = "gpt-4o-mini"

# For captions
system_prompt_captions = '''
You specialize in generating captions for objects in images.

Each object is labeled with a numeric ID and outlined in color. You'll be provided a list of IDs and object names in the format: ["1: name1", "2: name2", "3: name3", ...]. These names may be inaccurate.

Your task: Provide a Python list of dictionaries, where each dictionary includes the id, name, and a concise caption describing the object. Example format:
[
    {"caption": "brief description of object1"},
    {"caption": "brief description of object2"},
    ...
]
'''

system_prompt_consolidate_captions = '''
For each line, you will be provided with one idx and several names. Your task is to retrieve the most frequently mentioned name from each line and then determine its "category" with three rules:

1. If the object is a furniture typically placed on the floor and can support other objects or provide storage, label it as "category": "Asset". Do not include objects that are part of the furniture itself. Furniture is defined as large, functional items like tables, cabinets, or sofas, not small portable containers or decorations like baskets, vases, or lamps.
2. For objects that are suspended from the ceiling and those embedded in the wall, label it as "category": "Standalone".
3. For other objects, label it as "category": "Ordinary".

Your response should be in the format of a list of dictionaries, where each dictionary contains the "idx", "name", "category" of the object. Refer to the example below:

Example input:
[
    {"idx": 1, "names" = "sofa chair, couch, sofa chair, sofa chair, sofa chair, sofa chair, sofa chair, sofa chair, sofa chair"},
    {"idx": 2, "names" = "stool, stool, stool, stool, ottoman, stool, ottoman, stool, stool, stool"},
    {"idx": 3, "names" = "coffee kettle, coffee maker, coffee kettle, coffee kettle, coffee kettle, coffee kettle, coffee kettle, coffee kettle, coffee maker, coffee kettle"}
    ...
]

Example output:
[
    {"idx": 1, "name": "sofa chair", "category": "Asset"},
    {"idx": 2, "name": "stool", "category": "Standalone"},
    {"idx": 3, "name": "coffee kettle", "category": "Ordinary"}
    ...
]
'''

query_with_llm_prompts = '''
You will be given an list of objects in the room, your need to output top three objects' idx based on the relevance of the query.
Your response should be in the format of a list of dictionaries, where each dictionary contains the idx, name of the object. Refer to the example below:

The query you receive is a string, like this:
"Something to add light into the room."

And here is an example of the list of objects in the room:
[
    {"idx": "1", "name": "desk", "support": "Asset"},
    {"idx": "2", "name": "pillow", "support": "Ordinary"},
    {"idx": "3", "name": "power letout", "support": "Standalone"},
    ...
]

Example output:
[
    {"idx": "1", "name": "desk"},
    {"idx": "2", "name": "pillow"},
    {"idx": "3", "name": "power letout"},
]
'''

descriptive_query_with_llm_prompts = '''
You will be provided with a list of objects in the room.

Given a query, your task is to identify the target objects, which are placed on top of other objects. And then output top three objects based on similarity to the target object.

You only need to output the final result, and your response should be a list of dictionaries, each containing the 'idx', 'name' of the object. Refer to the example below:

The query you receive is a string, like this:
"This is a kettle on the table."

And here is an example of the list of objects in the room:
[
    {"idx": "1", "name": "couch", "support": "Asset", "descriptive": "A large, comfortable"},
    {"idx": "2", "name": "tissue box", "support": "Ordinary", "descriptive": "A box of tissues"},
    {"idx": "3", "name": "power outlet", "support": "Standalone", "descriptive": "A power outlet"},
    {"idx": "4", "name": "desk", "support": "Asset", "descriptive": "A wooden desk"},
    {"idx": "5", "name": "kettle", "support": "Standalone", "descriptive": "A metal kettle"},
    {"idx": "6", "name": "vase", "support": "Standalone", "descriptive": "A glass vase"},
    {"idx": "7", "name": "sofa", "support": "Asset", "descriptive": "A large, comfortable sofa"},
    {"idx": "9", "name": "arm chair", "support": "Asset", "descriptive": "A comfortable arm chair"},
    ...
]

Example output:
[
    {"idx": "5", "name": "kettle"},
    {"idx": "6", "name": "vase"},
    {"idx": "2", "name": "tissue box"},
]
'''

descriptive_query_with_llm_and_scene_graph_prompts = '''
You will be provided with a list of objects in the room and a dictionary representing the hierarchical relationships between these objects. Each 'key' in the dictionary represents a supporting object, and its 'value' is a list of objects placed on top of it. 

Given a query, your task is to identify the target objects, those placed on top of other objects. And then output top three objects based on similarity to the target object, reference to the object list and hierarchical relationships. If a 'key' does not contain the required 'value', search for the corresponding 'value' in other 'keys'. If no exact match is found in the hierarchical relationships, select the most likely target objects based on contextual relevance in the object list.

Your response should be a list of dictionaries, each containing the 'idx', 'name' of the object. Refer to the example below:

The query you receive is a string, like this:
"This is a pillow on the couch."

And here is an example of the list of objects in the room:
[
    {"idx": "1", "name": "couch", "category": "Asset"},
    {"idx": "2", "name": "pillow", "category": "Ordinary"},
    {"idx": "3", "name": "power outlet", "category": "Standalone"},
    {"idx": "4", "name": "cabinet", "category": "Asset"},
    {"idx": "5", "name": "coffee kettle", "category": "Ordinary"},
    {"idx": "6", "name": "pillow", "category": "Ordinary"},
    {"idx": "7", "name": "pillow", "category": "Ordinary"},
    {"idx": "8", "name": "table", "category": "Asset"},
    {"idx": "9", "name": "book", "category": "Ordinary"}
    ...
]

And here is an example of the dictionary regarding the hierarchy of objects:
{
    "1": ["2", "6", "7"],
    "4": ["5"],
    "8": ["9"],
    ...
}

Example output:
[
    {"idx": "2", "name": "pillow"},
    {"idx": "6", "name": "pillow"},
    {"idx": "7", "name": "pillow"},
]
'''

