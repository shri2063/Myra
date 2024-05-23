import streamlit as st
import os
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
CUTOUT_RANGE = {'left_up', 'left_low', 'backbone', 'right_up', 'right_low'}


def remove_file():
    if (os.path.exists('myra-app-main/predict/images/out_image.jpg')):
        os.remove('myra-app-main/predict/images/out_image.jpg')
    if (os.path.exists('myra-app-main/predict/images/out_image_transit.jpg')):
        os.remove('myra-app-main/predict/images/out_image_transit.jpg')
    if (os.path.exists('myra-app-main/predict/images/pg_output.png')):
        os.remove('myra-app-main/predict/images/pg_output.png')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_backbone.jpg')):
        os.remove('myra-app-main/predict/images/cutouts/im_backbone.jpg')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_left_low.jpg')):
        os.remove('myra-app-main/predict/images/cutouts/im_left_low.jpg')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_left_up.jpg')):
        os.remove('myra-app-main/predict/images/cutouts/im_left_up.jpg')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_right_low.jpg')):
        os.remove('myra-app-main/predict/images/cutouts/im_right_low.jpg')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_right_up.jpg')):
        os.remove('myra-app-main/predict/images/cutouts/im_right_up.jpg')
    if (os.path.exists('myra-app-main/predict/images/cutouts/im_right_up.jpg')):
        os.remove('myra-app-main/predict/images/cutouts/im_right_up.jpg')

#remove_file()

