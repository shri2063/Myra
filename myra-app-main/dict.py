import streamlit as st
import os
from PIL import Image
# Define the scope for Google Drive API




# API Tokens and endpoints from `.streamlit/secrets.toml` file
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]

# Resources text, link, and logo
replicate_text = "Stability AI SDXL Model on Replicate"
replicate_link = "https://replicate.com/stability-ai/sdxl"
replicate_logo = "https://storage.googleapis.com/llama2_release/Screen%20Shot%202023-07-21%20at%2012.34.05%20PM.png"

# Placeholders for myra_v1 and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()





# paste cloth
backbone = [4, 3, 2, 1, 0, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31]
left_up = [5, 6, 7, 12, 13, 14]
left_low = [7, 8, 9, 10, 11, 12]
right_up = [22, 23, 24, 29, 30, 31]
right_low = [24, 25, 26, 27, 28, 29]

CUTOUT_MAPPING = {
    'backbone': backbone,
    'left_up': left_up,
    'left_low': left_low,
    'right_up': right_up,
    'right_low': right_low
}
CUTOUT_RANGE = ['backbone', 'left_up', 'left_low', 'right_up', 'right_low']
BACK_IMAGES = ['model_image', 'out_image', 'uploaded_image', 'parse', "skin"]
FRONT_IMAGES = ['model_image', 'out_image', 'uploaded_image', 'parse', "skin"]



def remove_file():
    if (os.path.exists('myra-app-main/predict/images/out_image.png')):
        os.remove('myra-app-main/predict/images/out_image.png')
    if (os.path.exists('myra-app-main/predict/images/out_image_transit.png')):
        os.remove('myra-app-main/predict/images/out_image_transit.png')
    if (os.path.exists('myra-app-main/predict/images/pg_output.png')):
        os.remove('myra-app-main/predict/images/pg_output.png')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_backbone.png')):
        os.remove('myra-app-main/predict/images/cutouts/im_backbone.png')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_left_low.png')):
        os.remove('myra-app-main/predict/images/cutouts/im_left_low.png')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_left_up.png')):
        os.remove('myra-app-main/predict/images/cutouts/im_left_up.png')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_right_low.png')):
        os.remove('myra-app-main/predict/images/cutouts/im_right_low.png')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_right_up.png')):
        os.remove('myra-app-main/predict/images/cutouts/im_right_up.png')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_right_up.png')):
        os.remove('myra-app-main/predict/images/cutouts/im_right_up.png')
    if (os.path.exists('myra-app-main/predict/images/out_image_l_mask_ag.png')):
        os.remove('myra-app-main/predict/images/out_image_l_mask_ag.png')
    if (os.path.exists('myra-app-main/predict/images/out_mask.png')):
        os.remove('myra-app-main/predict/images/out_mask.png')
    if (os.path.exists('myra-app-main/predict/images/out_mask_l_mask_ag.png')):
        os.remove('myra-app-main/predict/images/out_mask_l_mask_ag.png')
    if (os.path.exists('myra-app-main/predict/images/skin_image.png')):
        os.remove('myra-app-main/predict/images/skin_image.png')

def remove_files_out_image():
    if (os.path.exists('myra-app-main/predict/images/out_image.png')):
        os.remove('myra-app-main/predict/images/out_image.png')
    if (os.path.exists('myra-app-main/predict/images/out_image_transit.png')):
        os.remove('myra-app-main/predict/images/out_image_transit.png')
    if (os.path.exists('myra-app-main/predict/images/pg_output.png')):
        os.remove('myra-app-main/predict/images/pg_output.png')
    if (os.path.exists('myra-app-main/predict/images/out_image_l_mask_ag.png')):
        os.remove('myra-app-main/predict/images/out_image_l_mask_ag.png')
    if (os.path.exists('myra-app-main/predict/images/out_mask.png')):
        os.remove('myra-app-main/predict/images/out_mask.png')
    if (os.path.exists('myra-app-main/predict/images/out_mask_l_mask_ag.png')):
        os.remove('myra-app-main/predict/images/out_mask_l_mask_ag.png')

# Define the colormap for the labels
COLORMAP = {
            1: [255, 0, 0],  # Red
            2: [0, 255, 0],  # Green
            3: [0, 0, 255],  # Blue
            4: [255, 255, 0],  # Yellow
            5: [255, 0, 255],  # Magenta
            6: [0, 255, 255]  # Cyan
                }

