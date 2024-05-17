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



### REMOVE FILES
if os.path.exists('myra-app-main/out_image.jpg'):
    os.remove('myra-app-main/out_image.jpg')
if os.path.exists('myra-app-main/out_mask.jpg'):
    os.remove('myra-app-main/out_mask.jpg')
if os.path.exists('myra-app-main/pg_output.png'):
    os.remove('myra-app-main/pg_output.png')

 # paste cloth
group_backbone = [4, 3, 2, 1, 0, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31]
group_left_up = [5, 6, 7, 12, 13, 14]
group_left_low = [7, 8, 9, 10, 11, 12]
group_right_up = [22, 23, 24, 29, 30, 31]
group_right_low = [24, 25, 26, 27, 28, 29]