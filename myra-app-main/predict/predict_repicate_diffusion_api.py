import numpy as np
import json
import replicate
import sys
sys.path.append("myra-app-main/upload_images")
import cloudinary_upload
import streamlit as st
import base64
# API Tokens and endpoints from `.streamlit/secrets.toml` file
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]

# Resources text, link, and logo
replicate_text = "Stability AI SDXL Model on Replicate"
replicate_link = "https://replicate.com/stability-ai/sdxl"
replicate_logo = "https://storage.googleapis.com/llama2_release/Screen%20Shot%202023-07-21%20at%2012.34.05%20PM.png"

s_pos_json = "myra-app-main/upload_images/openpose_json.json"
c_pos_json = "myra-app-main/upload_images/cloth-landmark-json.json"

IMAGE_ADDRESS = "myra-app-main/upload_images/image.jpg"
CLOTH_ADDRESS = "myra-app-main/upload_images/cloth.jpg"
AG_MASK_ADDRESS = "myra-app-main/upload_images/ag_mask.png"
SKIN_MASK_ADDRESS = "myra-app-main/upload_images/skin_mask.png"
PARSE_ADDRESS = "myra-app-main/upload_images/parse.png"
PARSE_AG_ADDRESS = "myra-app-main/upload_images/parse_ag_full.png"

def get_s_pos_string(s_pos_json) -> str:
    with open(s_pos_json, 'r') as f:
        s_pos = np.array(json.load(f)["people"][0]["pose_keypoints_2d"])
        s_pos = s_pos.tostring()
        s_pos = base64.b64encode(s_pos).decode('utf-8')
        return s_pos


def get_c_pos_string(c_pos_json) -> str:
    with open(c_pos_json, 'r') as f:
        c_pos = json.load(f)
        c_pos = np.array(c_pos["long"])
        c_pos = c_pos.tostring()
        c_pos = base64.b64encode(c_pos).decode('utf-8')
        return c_pos


image = cloudinary_upload.uploadImage(IMAGE_ADDRESS, 'image_1')
cloth = cloudinary_upload.uploadImage(CLOTH_ADDRESS, 'cloth_1')
ag_mask = cloudinary_upload.uploadImage(AG_MASK_ADDRESS, 'ag_mask_1')
skin_mask = cloudinary_upload.uploadImage(SKIN_MASK_ADDRESS, 'skin_mask_1')
parse = cloudinary_upload.uploadImage(PARSE_ADDRESS, 'parse_1')
parse_ag = cloudinary_upload.uploadImage(PARSE_AG_ADDRESS, 'parse_ag-1')
print("image", image)

output = replicate.run(
    "shrikantbhole/diffusion3:bc1f239ec073e1cfe92a13bcba6ce4b863e1852592612dbfabdb8039012e1807",
            input={
                "image": image,
                "cloth": cloth,
                "ag_mask": ag_mask,
                "skin_mask":skin_mask ,
                "parse": parse,
                "parse_ag" : parse_ag,
                "s_pos": get_s_pos_string(s_pos_json),
                "c_pos": get_c_pos_string(c_pos_json)
            }
        )

