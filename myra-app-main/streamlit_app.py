import streamlit as st
from streamlit_image_select import image_select
from PIL import Image, ImageDraw, ImageFont
import replicate
import requests
from io import BytesIO
from upload_images import cloudinary_upload
import os
from torchvision import transforms
import torch
import numpy as np
import json
import base64
from os import listdir
from math import ceil
import pandas as pd
import random
import string
from utility import  *
from streamlit_image_coordinates import streamlit_image_coordinates
from predict.predict_tps import generate_tps_st
from predict import predict_parse_seg_image as pg
from predict import predict_pos_keypoints as kg



# Define the scope for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']
st.set_page_config(page_title="Myra",
                   page_icon=":bridge_at_night:",
                   layout="wide")
st.markdown("# :rainbow[Myra - Your AI Creative Studio]")

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



if os.path.exists('myra-app-main/out_image.jpg'):
    os.remove('myra-app-main/out_image.jpg')
if os.path.exists('myra-app-main/out_mask.jpg'):
    os.remove('myra-app-main/out_mask.jpg')
if os.path.exists('myra-app-main/pg_output.png'):
    os.remove('myra-app-main/pg_output.png')

### Session State variables
if 'cover_area_pointer_list_tshirt' not in st.session_state:
    st.session_state.cover_area_pointer_list_tshirt = []
if 'cover_area_label_list_tshirt' not in st.session_state:
    st.session_state.cover_area_label_list_tshirt = []
if 'cover_area_pointer_list_model' not in st.session_state:
    st.session_state.cover_area_pointer_list_model = []
if 'cover_area_label_list_model' not in st.session_state:
    st.session_state.cover_area_label_list_model = []
if 'key_points_tshirt' not in st.session_state:
    st.session_state.key_points_tshirt = None
if 'key_points_model' not in st.session_state:
    st.session_state.key_points_model = None
if 'pg_output' not in st.session_state:
    st.session_state.pg_output = None
if 'point_selected' not in st.session_state:
    st.session_state.point_selected = {"x": 0, "y": 0}

def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application, 
    including the form for user inputs and the resources section.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Attention! Look here for warning ↑**", icon="👋🏾")

            with st.expander(":rainbow[**Refine your output here**]"):
                '''
                # Advanced Settings (for the curious minds!)
                width = st.number_input("Width of output image", value=1024)
                height = st.number_input("Height of output image", value=1024)
                num_outputs = st.slider(
                    "Number of myra_v1 to output", value=1, min_value=1, max_value=4)
                scheduler = st.selectbox('Scheduler', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                       'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
                num_inference_steps = st.slider(
                    "Number of denoising steps", value=50, min_value=1, max_value=500)
                guidance_scale = st.slider(
                    "Scale for classifier-free guidance", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
                prompt_strength = st.slider(
                    "Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of infomation in image)",
                    value=0.8, max_value=1.0, step=0.1)
                refine = st.selectbox(
                    "Select refine style to use (left out the other 2)", ("expert_ensemble_refiner", "None"))
                high_noise_frac = st.slider(
                    "Fraction of noise to use for `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
                
            prompt = st.text_area(
                ":orange[**Enter prompt: start typing, Shakespeare ✍🏾**]",
                value="An astronaut riding a rainbow unicorn, cinematic, dramatic")
            negative_prompt = st.text_area(":orange[**Party poopers you don't want in image? 🙅🏽‍♂️**]",
                                           value="the absolute worst quality, distorted features",
                                           help="This is a negative prompt, basically type what you don't want to see in the generated image")

            '''
            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)
            show_popup = st.sidebar.button("Show Pop-up")

        # Credits and resources
        st.divider()
        st.markdown(
            ":orange[**Resources:**]  \n"
            f"<img src='{replicate_logo}' style='height: 1em'> [{replicate_text}]({replicate_link})",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            ---
            Follow me on:

            𝕏 → [@tonykipkemboi](https://twitter.com/tonykipkemboi)

            LinkedIn → [Tony Kipkemboi](https://www.linkedin.com/in/tonykipkemboi)

            """
        )

        # return submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt
        return;


### Session State variables
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = '00035_00'
if 'selected_tshirt' not in st.session_state:
    st.session_state.selected_tshirt = '00034_00'


def update_point_over_image(edited_image: Image, node: str, value: dict, kp_arr: np.ndarray,
                            cover_area_pointer_list: list, cover_area_label_list: list,address:str):
    st.write("heyy")
    point_size = 5
    label_text = 'Point'
    draw = ImageDraw.Draw(edited_image)
    font_size = 16
    font = ImageFont.truetype("arial.ttf", font_size)
    text_width, text_height = 5, 5

    node = int(node)
    st.write(f"values earlier: Node: {node}--- {kp_arr[node][0]}--{kp_arr[node][1]}")
    kp_arr[node][0] = value["x"]
    kp_arr[node][1] = value["y"]
    st.write(f"values after {kp_arr[node][0]}--{kp_arr[node][1]}")

    original_image = Image.open(address)
    image_crop = original_image.crop(cover_area_label_list[node])
    edited_image.paste(image_crop, cover_area_label_list[node])
    image_crop = original_image.crop(cover_area_pointer_list[node])
    edited_image.paste(image_crop, cover_area_pointer_list[node])
    draw.ellipse((kp_arr[node][0] - point_size,
                  kp_arr[node][1] - point_size,
                  kp_arr[node][0] + point_size,
                  kp_arr[node][1] + point_size),
                 fill='red')
    cover_area_pointer = (int(kp_arr[node][0]) - int(point_size),
                          int(kp_arr[node][1]) - int(point_size),
                          int(kp_arr[node][0]) + int(point_size) + 5,
                          int(kp_arr[node][1]) + int(point_size) + 5)
    cover_area_pointer_list[node] = cover_area_pointer
    text_x = kp_arr[node][0] + point_size + 5  # Adjust for spacing
    text_y = kp_arr[node][1] - text_height // 2
    cover_area_label = (int(kp_arr[node][0] + point_size + 5),
                        int(kp_arr[node][1] - text_height // 2),
                        int(kp_arr[node][0]) + int(point_size) + 5 + int(
                            text_width),
                        int(kp_arr[node][1]) - text_height // 2 + int(
                            text_height))
    cover_area_label_list[node] = cover_area_label
    draw.text((text_x, text_y), str(node), fill='red', font=font)

def write_cover_areas_for_pointer_and_labels(arr: np.ndarray, image: Image, cover_area_pointer_list: list,
                                             cover_area_label_list: list):
    point_size = 5
    label_text = 'Point'

    draw = ImageDraw.Draw(image)
    font_size = 16
    font = ImageFont.truetype("arial.ttf", font_size)

    text_width, text_height = 5, 5

    # Define the font size and font
    for id, point in enumerate(arr):
        cover_area_pointer = (int(point[0]) - int(point_size), int(point[1]) - int(point_size),
                              int(point[0]) + int(point_size) + 5, int(point[1]) + int(point_size) + 5)
        cover_area_pointer_list.append(cover_area_pointer)

        cover_area_label = (int(point[0] + point_size + 5), int(point[1] - text_height // 2),
                            int(point[0]) + int(point_size) + 5 + int(text_width),
                            int(point[1]) - text_height // 2 + int(text_height))

        cover_area_label_list.append(cover_area_label)


def write_points_and_labels_over_image(arr: np.ndarray, image: Image) -> Image:
    point_size = 5
    label_text = 'Point'
    draw = ImageDraw.Draw(image)
    font_size = 16
    font = ImageFont.truetype("arial.ttf", font_size)
    text_width, text_height = 5, 5
    point_color = 'green'

    for id, point in enumerate(arr):
        draw.ellipse(
            (point[0] - point_size, point[1] - point_size, point[0] + point_size, point[1] + point_size),
            fill=point_color)
        text_x = point[0] + point_size + 5  # Adjust for spacing
        text_y = point[1] - text_height // 2
        draw.text((text_x, text_y), str(id), fill='green', font=font)
    return image


def main_page(AG_MASK_ADDRESS=None, SKIN_MASK_ADDRESS=None) -> None:
    """Main page layout and logic for generating myra_v1."""

    directory_models = r'myra-app-main/data/image'
    directory_tshirts = r'myra-app-main/data/cloth'



    def initialize(directory, type):

        if type == "model":
            if 'df_models' not in st.session_state:
                files = listdir(directory_models)
                df_models = pd.DataFrame({'file': files,
                                   'selected': [False] * len(files),
                                   'label': [''] * len(files)})
                df_models.set_index('file', inplace=True)
                st.session_state.df_models = df_models
                return df_models
        elif type == "tshirt":
            if 'df_tshirts' not in st.session_state:
                files = listdir(directory_tshirts)
                df_tshirts = pd.DataFrame({'file': files,
                                          'selected': [False] * len(files),
                                          'label': [''] * len(files)})
                df_tshirts.set_index('file', inplace=True)
                st.session_state.df_tshirts = df_tshirts
                return df_tshirts


    def update(image, type):
        if type == 'model':
            st.write(f'Model: {image[:8]}')
            st.session_state.selected_model = image[:8]
        if type == 'tshirt':
            st.write(f'Tshirt: {image[:8]}')
            st.session_state.selected_tshirt = image[:8]




    initialize( r'myra-app-main/data/image','model')
    initialize(r'myra-app-main/data/cloth', 'tshirt')
    controls_models = st.columns(3)
    files_models = st.session_state.df_models.index.tolist()
    st.write("SELECT A MODEL")
    with controls_models[0]:
        batch_size = st.select_slider("Batch size:", range(3, 15, 3), key = "batch_model")
    with controls_models[1]:
        row_size = st.select_slider("Row size:", range(1, 6), value=5, key = "row_model")
    num_batches = ceil(len(files_models) / batch_size)
    with controls_models[2]:
        page = st.selectbox("Page", range(1, num_batches + 1), key = "page_model")


    #### Showig
    batch_models = files_models[(page - 1) * batch_size: page * batch_size]
    grid_models = st.columns(row_size)
    col = 0


    for image in batch_models:

        with grid_models[col]:
            st.image(f'{directory_models}\{image}', caption='bike')
            #print(st.session_state.df_models.at[0,'selected'])
            #st.write(st.session_state.df_models.at[0,'selected'])
            st.checkbox("SELECT", key=f'image_{image}',
                        value=False,
                        on_change=update, args=(image, 'model'))


        col = (col + 1) % row_size

    controls_tshirts = st.columns(3)
    files_tshirts = st.session_state.df_tshirts.index.tolist()
    st.write("SELECT A TSHIRT")
    with controls_tshirts[0]:
        batch_size = st.select_slider("Batch size:", range(3, 15, 3), key = "batch_tshirt")
    with controls_tshirts[1]:
        row_size = st.select_slider("Row size:", range(1, 6), value=5, key = "row_tshirt")
    num_batches = ceil(len(files_tshirts) / batch_size)
    with controls_tshirts[2]:
        page = st.selectbox("Page", range(1, num_batches + 1), key = "page_tshirt")

    grid_tshirts = st.columns(row_size)
    col = 0
    batch_tshirts = files_tshirts[(page - 1) * batch_size: page * batch_size]
    for image in batch_tshirts:

        with grid_tshirts[col]:
            st.image(f'{directory_tshirts}\{image}', caption='bike')
            #print(st.session_state.df_models.at[0,'selected'])
            #st.write(st.session_state.df_models.at[0,'selected'])
            st.checkbox("SELECT", key=f'tshirt_{image}',
                        value=False,
                        on_change=update, args=(image, 'tshirt'))


        col = (col + 1) % row_size









    # Create two columns to show cloth and model Image
    col1, col2 = st.columns(2)

    # Display the myra_v1 in the column along with keypoints

    with col1:
        st.write("SELECTED MODEL")
        st.image(f'myra-app-main/data/image/{st.session_state.selected_model}.jpg')


    with col2:
        st.write("SELECTED TSHIRT")
        st.image(f'myra-app-main/data/cloth/{st.session_state.selected_tshirt}.jpg')


    # Create a button

    button_clicked = st.button("Run Diffusion!")


    # Check if the button is clicked

    s_pos_json = f'myra-app-main/data/openpose_json/{st.session_state.selected_model}_keypoints.json'
    c_pos_json = f'myra-app-main/data/cloth-landmark-json/{st.session_state.selected_tshirt}.json'

    IMAGE_ADDRESS = f'myra-app-main/data/image/{st.session_state.selected_model}.jpg'
    CLOTH_ADDRESS = f'myra-app-main/data/cloth/{st.session_state.selected_tshirt}.jpg'
    AG_MASK_ADDRESS = f'myra-app-main/data/ag_mask/{st.session_state.selected_model}.png'
    SKIN_MASK_ADDRESS = f'myra-app-main/data/skin_mask/{st.session_state.selected_model}.png'
    PARSE_ADDRESS = f'myra-app-main/data/parse/{st.session_state.selected_model}.png'
    PARSE_AG_ADDRESS = f'myra-app-main/data/parse_ag/{st.session_state.selected_model}.png'

    def generate_random_string(length):
        # Choose from all uppercase letters and digits
        characters = string.ascii_uppercase + string.digits
        # Generate a random string of given length
        return ''.join(random.choices(characters, k=length))

    image_str = generate_random_string(12)
    cloth_str = generate_random_string(12)
    ag_mask_str = generate_random_string(12)
    skin_mask_str = generate_random_string(12)
    parse_str = generate_random_string(12)
    parse_ag_str = generate_random_string(12)













    if button_clicked:
        st.write("Button clicked!")

        image = cloudinary_upload.uploadImage(IMAGE_ADDRESS, image_str)
        cloth = cloudinary_upload.uploadImage(CLOTH_ADDRESS, cloth_str)
        ag_mask = cloudinary_upload.uploadImage(AG_MASK_ADDRESS, ag_mask_str)
        skin_mask = cloudinary_upload.uploadImage(SKIN_MASK_ADDRESS, skin_mask_str)
        parse = cloudinary_upload.uploadImage(PARSE_ADDRESS, parse_str)
        parse_ag = cloudinary_upload.uploadImage(PARSE_AG_ADDRESS, parse_ag_str)


        output = replicate.run(
            "shrikantbhole/diffusion3:bc1f239ec073e1cfe92a13bcba6ce4b863e1852592612dbfabdb8039012e1807",
            input={
                "image": image,
                "cloth": cloth,
                "ag_mask": ag_mask,
                "skin_mask": skin_mask,
                "parse": parse,
                "parse_ag": parse_ag,
                "s_pos": get_s_pos_string(s_pos_json),
                "c_pos": get_c_pos_string(c_pos_json)
            }
        )

        print(output)
        response = requests.get(output)
        final_image = np.array(Image.open(BytesIO(response.content)))
        st.image(final_image, caption='final Image', use_column_width=True)

    #############PLOT KEYPOINTS OVER TSHIRT#############

    # Create two columns to show cloth and model Image
    col1, col2 = st.columns(2)

    # Display the myra_v1 in the column along with keypoints

    with col1:
            # Create an input text box to select a keypoint whose position needs to be changed


            node = st.text_input('Enter node position to change')
            if node:
                st.write("You are modifying Node " + node + "   Please click on new position")

            if os.path.exists(CLOTH_ADDRESS):

                with open(c_pos_json, 'r') as file:


                    json_list = json.load(file)
                    kp_arr = np.array(json_list["long"]) * 250
                    st.session_state.key_points_tshirt = kp_arr

                
                image = Image.open(CLOTH_ADDRESS)

                if not st.session_state.cover_area_pointer_list_tshirt:
                    write_cover_areas_for_pointer_and_labels(kp_arr, image,
                                                             st.session_state.cover_area_pointer_list_tshirt,
                                                             st.session_state.cover_area_label_list_tshirt)


                write_points_and_labels_over_image(kp_arr, image)

                ## Streamlit Image coordinate is a spl library in streamlit that captures point coordinates
                ## of a pixel clicked by mouse over the image
                value = streamlit_image_coordinates(
                    image,
                    key="pil",
                )

                if value and (value["x"] != st.session_state.point_selected["x"] and value["y"] !=
                              st.session_state.point_selected["y"]):

                    st.session_state.point_selected = value
                    if node:

                        update_point_over_image(image, node, value, st.session_state.key_points_tshirt,
                                                st.session_state.cover_area_pointer_list_tshirt,
                                                st.session_state.cover_area_label_list_tshirt,CLOTH_ADDRESS)


                    else:
                        st.sidebar.write("Please select a node first")

                    # out_image = cloudinary_upload.uploadImage('myra-app-main/upload_images/image.jpg', 'tshirt')

    with col2:
            image = Image.open(CLOTH_ADDRESS)
            write_points_and_labels_over_image( st.session_state.key_points_tshirt, image)
            st.image(image, use_column_width=True)

            ###################KEYPOINT DETECTOR#########################
            # Create two columns to show cloth and model Image
    col1, col2 = st.columns(2)

    # Display the myra_v1 in the columns

    with col1:

        model_image = Image.open(IMAGE_ADDRESS)
        if st.session_state.key_points_model is not None:
            write_points_and_labels_over_image(st.session_state.key_points_model, model_image)

        # If Key Point Detector Is called
        key_point_detector = st.button("Run KeyPoint Detector!")
        if key_point_detector:


            key_points = kp_arr


            p_pos = get_p_pos(key_points, s_pos_json)

            st.session_state.key_points_model = p_pos

            model_image = Image.open(IMAGE_ADDRESS)
            write_points_and_labels_over_image(p_pos, model_image)
            st.session_state.cover_area_pointer_list_model = []
            st.session_state.cover_area_label_list_model = []
            write_cover_areas_for_pointer_and_labels(st.session_state.key_points_model, model_image,
                                                     st.session_state.cover_area_pointer_list_model,
                                                     st.session_state.cover_area_label_list_model)

        model_node = st.text_input('Enter node position to change', key='model_node')
        if model_node:
            st.write("You are modifying Node " + str(model_node) + "   Please click on new position")
            # Create a button

        model_value = streamlit_image_coordinates(
            model_image,
            key="model_value",
        )

        if model_value and (model_value["x"] != st.session_state.point_selected["x"] and model_value["y"] !=
                            st.session_state.point_selected["y"]):


            st.session_state.point_selected = model_value
            if model_node:
                st.write("heyy")
                update_point_over_image(model_image, model_node, model_value, st.session_state.key_points_model,
                                        st.session_state.cover_area_pointer_list_model,
                                        st.session_state.cover_area_label_list_model,IMAGE_ADDRESS)



            else:
                st.sidebar.write("Please select a node first")

    with col2:

        model_image = Image.open(IMAGE_ADDRESS)
        write_points_and_labels_over_image(st.session_state.key_points_model, model_image)
        st.image(model_image, use_column_width=True)


    ###  PARSED IMAGE GENERATOR OUTPUT###

    parse_image_generator = st.button("Generate Parsed Image!")
    if parse_image_generator:
        model_parse_ag_full_image = Image.open(PARSE_AG_ADDRESS)
        p_pos = st.session_state.key_points_model.copy()
        st.write(f"values earlier:  {p_pos[1][0]}--{p_pos[1][1]}")
        ### temporary

        p_pos = torch.tensor(p_pos)
        p_pos[:, 0] = p_pos[:, 0] / 768
        p_pos[:, 1] = p_pos[:, 1] / 1024
        p_pos = p_pos.float()

        pg_output, parse13_model_seg = pg.parse_model_seg_image(get_s_pos(s_pos_json), p_pos.clone(), model_parse_ag_full_image)
        model_parse_gen_image = pg.draw_parse_model_image(pg_output)
        st.image(model_parse_gen_image)
        st.session_state.pg_output = pg_output
        #model_parse_gen_image = pg.draw_parse_model_image(pg_output)




    ###########Tshirt Wrapper PIPELINE######################

    tshirt_warp_generator = st.button("Generate Tshirt Warper!")
    if tshirt_warp_generator:
        p_pos = st.session_state.key_points_model.copy()
        p_pos = torch.tensor(p_pos)
        p_pos[:, 0] = p_pos[:, 0] / 768
        p_pos[:, 1] = p_pos[:, 1] / 1024
        p_pos = p_pos.float()

        # ag_mask = 255 - cv2.imread(AG_MASK_ADDRESS, cv2.IMREAD_GRAYSCALE)
        ag_mask = 255 - np.asarray(Image.open(AG_MASK_ADDRESS))

        ag_mask = np.array(transforms.Resize(768)(Image.fromarray(ag_mask)))
        skin_mask = np.asarray(Image.open(SKIN_MASK_ADDRESS).convert('L'))

        skin_mask = np.array(transforms.Resize(768)(Image.fromarray(skin_mask)))

        c_pos, v_pos = get_c_pos_warp(c_pos_json)

        out_image, out_mask = generate_tps_st(
            np.asarray(Image.open(IMAGE_ADDRESS)),
            np.asarray(Image.open(CLOTH_ADDRESS)),
            v_pos.float(),
            torch.tensor(p_pos),
            ag_mask,
            skin_mask,
            st.session_state.pg_output

        )
        st.image(Image.fromarray(out_image))
        Image.fromarray(out_image).save('myra-app-main/predict/images/out_image.jpg')
        Image.fromarray(out_mask).save('myra-app-main/predict/images/out_mask.jpg')
        # st.image(Image.fromarray(out_mask))


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates myra_v1 based on these inputs.
    """
    # submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt = configure_sidebar()
    try:
        main_page()
    except Exception as e:
        1 == 1


if __name__ == "__main__":
    main()
    configure_sidebar()
