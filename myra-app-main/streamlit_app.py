import os
from io import BytesIO

import numpy as np
import requests
import streamlit as st
import replicate
## Set Page framework
st.set_page_config(page_title="Myra",
                   page_icon=":bridge_at_night:",
                   layout="wide")
st.markdown("# :rainbow[Myra - Your AI Creative Studio]")

from upload_images import cloudinary_upload

from math import ceil

from utility import *
from streamlit_image_coordinates import streamlit_image_coordinates
from predict import predict_parse_seg_image as pg
from dict import *
from tps_services import *


import smtplib
from email.mime.text import MIMEText


### Session State variables
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'token' not in st.session_state:
    st.session_state.token = 'blaa'
if 'key_points_tshirt' not in st.session_state:
    st.session_state.key_points_tshirt = None
if 'key_points_model' not in st.session_state:
    st.session_state.key_points_model = None
if 'pg_output' not in st.session_state:
    st.session_state.pg_output = None
if 'point_selected_model' not in st.session_state:
    st.session_state.point_selected_model = {"x": 0, "y": 0}
if 'point_selected_tshirt' not in st.session_state:
    st.session_state.point_selected_tshirt = {"x": 0, "y": 0}
if 'point_selected_cutout' not in st.session_state:
    st.session_state.point_selected_cutout = {"x": 0, "y": 0}
if 'point_selected_skin_mask' not in st.session_state:
    st.session_state.point_selected_skin_mask = {"x": 0, "y": 0}
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = '00035_00'
if 'selected_tshirt' not in st.session_state:
    st.session_state.selected_tshirt = '00067_00'
if 'selected_cutout' not in st.session_state:
    st.session_state.selected_cutout = None
if 'cutout_list' not in st.session_state:
    st.session_state.cutout_list = {}

### History
if 'last_selected_cutout_pt' not in st.session_state:
    st.session_state.last_selected_cutout_pt = 0
if 'last_selected_model_pt' not in st.session_state:
    st.session_state.last_selected_model_pt = {}
if 'last_selected_tshirt_pt' not in st.session_state:
    st.session_state.last_selected_tshirt_pt = {}


### SKIN MASKS
if 'skin_mask_selected_points' not in st.session_state:
    st.session_state.skin_mask_selected_points = []
if 'skin_mask_replace_color' not in st.session_state:
    st.session_state.skin_mask_replace_color = []
if 'select_colour' not in st.session_state:
    st.session_state.select_colour = False
if 'skin_replant_image_arr' not in st.session_state:
    st.session_state.skin_replant_image_arr = None
if 'skin_mask_replant_image_arr' not in st.session_state:
    st.session_state.skin_mask_replant_image_arr = None
if 'skin_mask_selected_points_history' not in st.session_state:
    st.session_state.skin_mask_selected_points_history = []


if 'selected_skin_mask' not in st.session_state:
    st.session_state.selected_skin_mask = "default"
if 'skin_out_image_toggle' not in st.session_state:
    st.session_state.skin_out_image_toggle = False

def home_page():
    st.write(
        "<span style='font-family: Roboto, sans-serif;'>At Myra, our vision is to inject a sense of magic and creativity into e-commerce fashion "
        "photoshoots by leveraging AI-generated image. </span>",
        unsafe_allow_html=True)
    st.write(" ")

    st.write(
        "<span style='font-family: Roboto, sans-serif;'> Here's the concept: We begin with an image of a mannequin showcasing the fashion product under ideal "
        "lighting conditions. This image, along with your prompt detailing the desired attributes of the final model, is fed into our Myra AI system. "
        "From there, Myra AI swiftly crafts the perfect image of real looking model tailored to your specifications in no time.</span>",
        unsafe_allow_html=True)

    st.write(
        "<span style='font-family: Roboto,sans-serif'>We have listed below few examples of results obtained from Myra AI .The mannequin home below were retrieved from the internet from different sites. "
        "pinterest, shutterstock, istockphoto. "
        " Myra AI could "
        "fit AI image within this dress, maintaining the dress outline, tone, and appearance.</span>",
        unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")


    #####  MODEL A #######

    image1 = Image.open("myra-app-main/home/MA/05700_00.jpg")
    image2 = Image.open("myra-app-main/home/MA/07262_00.jpg")
    image3 = Image.open("myra-app-main/home/MA/00375_F2.png")
    image4 = Image.open("myra-app-main/home/MA/00413_F2.png")
    image5 = Image.open("myra-app-main/home/MA/00540_F2.png")
    image6 = Image.open("myra-app-main/home/MA/07587_F2.png")

    # Create two columns for displaying home side by side

    col1, col2 = st.columns(2)

    # Display the home in the columns
    with col1:
        st.image(image1, caption='Input Image (Tshirt)', use_column_width=True)

    with col2:
        st.image(image2, caption='Input Image (Model)', use_column_width=True)
    st.write("")
    st.write("")
    st.write("")

    st.info("Now Let's brew  some Myra magic !!!")

    # Create two columns for displaying home side by side
    col3, col4 = st.columns(2)

    # Display the home in the columns
    with col3:
        st.image(image3, caption='Output Image', use_column_width=True)

    with col4:
        st.image(image4, caption='Output Image', use_column_width=True)


    # Create two columns for displaying home side by side
    col5, col6 = st.columns(2)

    # Display the home in the columns
    with col5:
        st.image(image5, caption='Output Image', use_column_width=True)

    with col6:
        st.image(image6, caption='Output Image', use_column_width=True)

    st.write("")
    st.write("")
    st.write("")

    #####  MODEL D #######
    image1 = Image.open("myra-app-main/home/MD/06758_00.jpg")
    image2 = Image.open("myra-app-main/home/MD/07046_00.jpg")
    image3 = Image.open("myra-app-main/home/MD/10930_03_F2.png")
    image4 = Image.open("myra-app-main/home/MD/04963_03_F1.png")
    image5 = Image.open("myra-app-main/home/MD/02364_00_F1.png")
    image6 = Image.open("myra-app-main/home/MD/09566_01_F2.png")

    # Create two columns for displaying home side by side

    col1, col2 = st.columns(2)

    # Display the home in the columns
    with col1:
        st.image(image1, caption='Input Image (Tshirt)', use_column_width=True)

    with col2:
        st.image(image2, caption='Input Image (Model)', use_column_width=True)
    st.write("")
    st.write("")
    st.write("")

    st.info("With Some Myra Magic following Model creatives can be created !!!")

    # Create two columns for displaying home side by side
    col3, col4 = st.columns(2)

    # Display the home in the columns
    with col3:
        st.image(image3, caption='Output Image', use_column_width=True)

    with col4:
        st.image(image4, caption='Output Image', use_column_width=True)

    # Create two columns for displaying home side by side
    col5, col6 = st.columns(2)

    # Display the home in the columns
    with col5:
        st.image(image5, caption='Output Image', use_column_width=True)

    with col6:
        st.image(image6, caption='Output Image', use_column_width=True)

    st.write("")
    st.write("")
    st.write("")

    #####  MODEL F #######
    image1 = Image.open("myra-app-main/home/MF/07025_00.jpg")
    image2 = Image.open("myra-app-main/home/MF/01994_00.jpg")
    image3 = Image.open("myra-app-main/home/MF/03159_03_F2.png")
    image4 = Image.open("myra-app-main/home/MF/01815_01_F5.png")
    image5 = Image.open("myra-app-main/home/MF/06442_01_F3.png")
    image6 = Image.open("myra-app-main/home/MF/09264_00_f2.png")

    # Create two columns for displaying home side by side

    col1, col2 = st.columns(2)

    # Display the home in the columns
    with col1:
        st.image(image1, caption='Input Image (Tshirt)', use_column_width=True)

    with col2:
        st.image(image2, caption='Input Image (Model)', use_column_width=True)
    st.write("")
    st.write("")
    st.write("")

    st.info("Here you go with the Myra AI!!")

    # Create two columns for displaying home side by side
    col3, col4 = st.columns(2)

    # Display the home in the columns
    with col3:
        st.image(image3, caption='Output Image', use_column_width=True)

    with col4:
        st.image(image4, caption='Output Image', use_column_width=True)

    # Create two columns for displaying home side by side
    col5, col6 = st.columns(2)

    # Display the home in the columns
    with col5:
        st.image(image5, caption='Output Image', use_column_width=True)

    with col6:
        st.image(image6, caption='Output Image', use_column_width=True)

def main_page(AG_MASK_ADDRESS=None, SKIN_MASK_ADDRESS=None) -> None:
    """Main page layout and logic for generating home."""

    s_pos_json = f'myra-app-main/data/openpose_json/{st.session_state.selected_model}_keypoints.json'
    c_pos_json = f'myra-app-main/data/cloth-landmark-json/{st.session_state.selected_tshirt}.json'
    IMAGE_ADDRESS = f'myra-app-main/data/image/{st.session_state.selected_model}.jpg'
    CLOTH_ADDRESS = f'myra-app-main/data/cloth/{st.session_state.selected_tshirt}.jpg'
    AG_MASK_ADDRESS = f'myra-app-main/data/ag_mask/{st.session_state.selected_model}.png'
    SKIN_MASK_ADDRESS = f'myra-app-main/data/skin_mask/{st.session_state.selected_model}.png'
    PARSE_ADDRESS = f'myra-app-main/data/parse/{st.session_state.selected_model}.png'
    PARSE_AG_ADDRESS = f'myra-app-main/data/parse_ag/{st.session_state.selected_model}.png'

    IMAGES_MAPPING = {
        'model_image': IMAGE_ADDRESS,
        'out_image': 'myra-app-main/predict/images/out_image.png',
        'parse':PARSE_ADDRESS,
        'skin': SKIN_MASK_ADDRESS
    }

    ######CREATE CATALOG############

    initialize('model')  ### initialize df_models sesion state
    initialize('tshirt')  ### initialize df_tshirt  sesion state
    controls_models = st.columns(3)
    files_models = st.session_state.df_models.index.tolist()

    def update(image, type):
        st.write("Update ")
        remove_file()
        st.session_state.key_points_model = None
        st.session_state.pg_output == None
        if type == 'model':
            st.session_state.selected_model = image[:8]
            st.session_state.key_points_model = None

        if type == 'tshirt':

            st.session_state.selected_tshirt = image[:8]
            st.session_state.key_points_tshirt = None

            if (os.path.exists('myra-app-main/predict/images/out_image.png')):
                os.remove('myra-app-main/predict/images/out_image.png')

    ### Select a Model
    st.write("SELECT A MODEL")
    with controls_models[0]:
        batch_size = st.select_slider("Batch size:", range(3, 15, 3), key="batch_model")
    with controls_models[1]:
        row_size = st.select_slider("Row size:", range(1, 6), value=5, key="row_model")
    num_batches = ceil(len(files_models) / batch_size)
    with controls_models[2]:
        page = st.selectbox("Page", range(1, num_batches + 1), key="page_model")

    batch_models = files_models[(page - 1) * batch_size: page * batch_size]
    grid_models = st.columns(row_size)
    col = 0

    for image in batch_models:
        with grid_models[col]:
            st.write()
            st.image(f'myra-app-main/data/image/{image}', caption=f'{image[:8]}')
            # print(st.session_state.df_models.at[0,'selected'])
            # st.write(st.session_state.df_models.at[0,'selected'])
            st.checkbox("SELECT", key=f'image_{image}',
                        value=False,
                        on_change=update, args=(image, 'model'))

        col = (col + 1) % row_size

    controls_tshirts = st.columns(3)
    files_tshirts = st.session_state.df_tshirts.index.tolist()

    ### Select a Tshirt
    st.write("SELECT A TSHIRT")
    with controls_tshirts[0]:
        batch_size = st.select_slider("Batch size:", range(3, 15, 3), key="batch_tshirt")
    with controls_tshirts[1]:
        row_size = st.select_slider("Row size:", range(1, 6), value=5, key="row_tshirt")
    num_batches = ceil(len(files_tshirts) / batch_size)
    with controls_tshirts[2]:
        page = st.selectbox("Page", range(1, num_batches + 1), key="page_tshirt")

    grid_tshirts = st.columns(row_size)
    col = 0
    batch_tshirts = files_tshirts[(page - 1) * batch_size: page * batch_size]
    for image in batch_tshirts:
        with grid_tshirts[col]:
            st.image(f'myra-app-main/data/cloth/{image}', caption=f'{image[:8]}')
            # print(st.session_state.df_models.at[0,'selected'])
            # st.write(st.session_state.df_models.at[0,'selected'])
            st.checkbox("SELECT", key=f'tshirt_{image}',
                        value=False,
                        on_change=update, args=(image, 'tshirt'))

        col = (col + 1) % row_size

    # Create two columns to show cloth and model Image
    col1, col2 = st.columns(2)

    # Display the home in the column along with keypoints

    with col1:
        st.write(f"SELECTED MODEL: {st.session_state.selected_model}")
        st.image(f'myra-app-main/data/image/{st.session_state.selected_model}.jpg')

    with col2:
        st.write(f"SELECTED TSHIRT: {st.session_state.selected_tshirt}")
        st.image(f'myra-app-main/data/cloth/{st.session_state.selected_tshirt}.jpg')

    #############PLOT KEYPOINTS OVER TSHIRT#############

    refresh = st.button("Refresh App")
    if refresh:
        remove_file()
    # Create an input text box to select a keypoint whose position needs to be changed
    st.warning("WARNING:  ONCE CLICKED ALL PREVIOUS KEPYPOINTS DATA WOULD BE ERASED")

    col1, col2 = st.columns(2)
    with col1:
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        node = st.text_input('Enter node position to change')
        if node:
            st.write("You are modifying Node " + node + "   Please click on new position")

        tshirt_kp_seg = st.selectbox("Keypoints Segments", CUTOUT_RANGE, index=None, key="tshirt_kp_seg")

        back_tshirt = st.button("Undo Last Edited Tshirt Point")
        if back_tshirt:
            tshirt_node_last = st.session_state.last_selected_tshirt_pt['node']
            x = st.session_state.last_selected_tshirt_pt['x']
            y = st.session_state.last_selected_tshirt_pt['y']

            st.session_state.key_points_tshirt[int(tshirt_node_last)][0] = x
            st.session_state.key_points_tshirt[int(tshirt_node_last)][1] = y


    with col2:


        if os.path.exists(CLOTH_ADDRESS):

            if st.session_state.key_points_tshirt is None:
                with open(c_pos_json, 'r') as file:
                    json_list = json.load(file)
                    kp_arr = np.array(json_list["long"]) * 250
                    st.session_state.key_points_tshirt = kp_arr

            tshirt_image = Image.open(CLOTH_ADDRESS)


            if tshirt_kp_seg != None:


                write_points_and_labels_over_image( st.session_state.key_points_tshirt[1:][CUTOUT_MAPPING[tshirt_kp_seg]],
                                                   tshirt_image, CUTOUT_MAPPING[tshirt_kp_seg])
            else:
                write_points_and_labels_over_image( st.session_state.key_points_tshirt[1:], tshirt_image, None)

            st.write("hi")
            ## Streamlit Image coordinate is a spl library in streamlit that captures point coordinates
            ## of a pixel clicked by mouse over the image
            value = streamlit_image_coordinates(
                tshirt_image,
                key="kpt",
            )

            if value and (value["x"] != st.session_state.point_selected_tshirt["x"] and value["y"] !=
                          st.session_state.point_selected_tshirt["y"]):


                st.session_state.point_selected_tshirt = value
                if node:

                    st.session_state.last_selected_tshirt_pt['node'] = int(node) + 1

                    st.session_state.last_selected_tshirt_pt['x'] = st.session_state.key_points_tshirt[int(node) + 1][0]

                    st.session_state.last_selected_tshirt_pt['y'] = st.session_state.key_points_tshirt[int(node) + 1][1]

                    st.write( st.session_state.key_points_tshirt[int(node) + 1])
                    st.session_state.key_points_tshirt[int(node) + 1][0] = value["x"]
                    st.session_state.key_points_tshirt[int(node) + 1][1] = value["y"]
                    st.write(st.session_state.key_points_tshirt[int(node)])
                    st.rerun()

                else:
                    st.sidebar.write("Please select a node first")







    ###################KEYPOINT DETECTOR#########################
    # Create two columns to show cloth and model Image

    model_image = Image.open(IMAGE_ADDRESS)
    if st.session_state.key_points_model is not None:
        write_points_and_labels_over_image(st.session_state.key_points_model, model_image)

    # If Key Point Detector Is called
    # Create two columns to show cloth and model Image
    st.warning("WARNING:  ONCE CLICKED ALL PREVIOUS KEPYPOINTS DATA WOULD BE ERASED")
    key_point_detector = st.button("Run KeyPoint Detector!")


    col1, col2 = st.columns(2)

    with col1:
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        model_node = st.text_input('Enter node position to change', key='model_node')
        if model_node:
                st.write("You are modifying Node " + str(model_node) + "   Please click on new position")



        model_kp_seg = st.selectbox("Keypoints Segments", CUTOUT_RANGE, index=None, key="model_kp_seg")




        model_back = st.button("Undo Last Edited Model Point")
        if model_back:

            model_node_last = st.session_state.last_selected_model_pt['model_node']
            x = st.session_state.last_selected_model_pt['x']
            y = st.session_state.last_selected_model_pt['y']

            st.session_state.key_points_model[int(model_node_last)][0] = x
            st.session_state.key_points_model[int(model_node_last)][1] = y



    with col2:


        if key_point_detector:
            key_points = st.session_state.key_points_tshirt
            st.session_state.key_points_model = get_p_pos(key_points, s_pos_json)

        model_image = Image.open(IMAGE_ADDRESS)

        if model_kp_seg != None:
            write_points_and_labels_over_image(st.session_state.key_points_model[CUTOUT_MAPPING[model_kp_seg]], model_image, CUTOUT_MAPPING[model_kp_seg])
        else:
            write_points_and_labels_over_image(st.session_state.key_points_model, model_image)




        model_value = streamlit_image_coordinates(
            model_image,
            key="kpd",
        )

        if model_value and (model_value["x"] != st.session_state.point_selected_model["x"] and model_value["y"] !=
                            st.session_state.point_selected_model["y"]):

            st.session_state.point_selected_model = model_value

            if model_node:
                st.session_state.last_selected_model_pt['model_node'] = model_node
                st.session_state.last_selected_model_pt['x'] =  st.session_state.key_points_model[int(model_node)][0]
                st.session_state.last_selected_model_pt['y'] =  st.session_state.key_points_model[int(model_node)][1]
                st.session_state.key_points_model[int(model_node)][0] = model_value["x"]
                st.session_state.key_points_model[int(model_node)][1] = model_value["y"]
                st.rerun()

            else:
                st.sidebar.write("Please select a node first")




    ###########Tshirt Wrapper PIPELINE######################

    generate_wrap = st.button("Generate Warps ")

    if generate_wrap:
        model_parse_ag_full_image = Image.open(PARSE_AG_ADDRESS)
        p_pos = st.session_state.key_points_model.copy()
        p_pos = torch.tensor(p_pos)
        p_pos[:, 0] = p_pos[:, 0] / 768
        p_pos[:, 1] = p_pos[:, 1] / 1024
        p_pos = p_pos.float()

        pg_output, parse13_model_seg = pg.parse_model_seg_image(get_s_pos(s_pos_json), p_pos.clone(),
                                                                model_parse_ag_full_image)

        # st.image(model_parse_gen_image)
        st.session_state.pg_output = pg_output

        ag_mask = 255 - np.asarray(Image.open(AG_MASK_ADDRESS))
        ag_mask = np.array(transforms.Resize(768)(Image.fromarray(ag_mask)))
        skin_mask = np.asarray(Image.open(SKIN_MASK_ADDRESS).convert('L'))
        skin_mask = np.array(transforms.Resize(768)(Image.fromarray(skin_mask)))
        c_pos, v_pos = get_c_pos_warp(st.session_state.key_points_tshirt.copy())

        generate_and_save_warp_images(
            np.asarray(Image.open(IMAGE_ADDRESS)),
            np.asarray(Image.open(CLOTH_ADDRESS)),
            v_pos.float(),
            torch.tensor(p_pos),
            ag_mask,
            skin_mask,
            st.session_state.pg_output

        )

        if st.session_state.cutout_list == {}:
            st.session_state.cutout_list['backbone'] = st.session_state.key_points_model[CUTOUT_MAPPING.get('backbone', [])]
            st.session_state.cutout_list['left_up'] = st.session_state.key_points_model[CUTOUT_MAPPING.get('left_up', [])]
            st.session_state.cutout_list['left_low'] = st.session_state.key_points_model[CUTOUT_MAPPING.get('left_low', [])]
            st.session_state.cutout_list['right_up'] = st.session_state.key_points_model[CUTOUT_MAPPING.get('right_up', [])]
            st.session_state.cutout_list['right_low'] = st.session_state.key_points_model[
            CUTOUT_MAPPING.get('right_low', [])]

    cutout_files = os.listdir('myra-app-main/predict/images/cutouts')
    if len(cutout_files) > 0 and st.session_state.key_points_model is not None:
        col1, col2 = st.columns(2)

        with col1:
            controls_models = st.columns(2)

            with controls_models[0]:
                wrap_blend_intensity = st.select_slider("Intensity", range(10, 100, 10), key="wrap_blend_intensity")

            with controls_models[1]:
                wrap_selected_cutout = st.selectbox("cutouts", CUTOUT_RANGE, index=0, key="wrap__dropdown")

            # Load the images
            model_image = Image.open(IMAGE_ADDRESS)

            wrap_cutout = [file for file in cutout_files if wrap_selected_cutout in file]
            cutout_image = np.asarray(Image.open(os.path.join('myra-app-main/predict/images/cutouts', wrap_cutout[0])))

            cutout_image_processed = process_cutout(cutout_image, np.asarray(model_image))

            # Define the weight for each image
            alpha = wrap_blend_intensity / 100  # Weight for the model image
            beta = (100 - wrap_blend_intensity) / 100  # Weight for the cutout image

            wrap_blended_image = cv2.addWeighted(np.asarray(model_image)[:, :, :3], alpha, cutout_image_processed, beta,
                                                 0)

            wrap_blended_image = Image.fromarray(wrap_blended_image.astype(np.uint8))

            write_points_and_labels_over_image(
                st.session_state.key_points_model[CUTOUT_MAPPING.get(wrap_selected_cutout, [])], wrap_blended_image,
                CUTOUT_MAPPING.get(wrap_selected_cutout, []))

            st.image(wrap_blended_image)

        with col2:
            if len(cutout_files) > 0:
                write_points_and_labels_over_image(
                    st.session_state.key_points_model[CUTOUT_MAPPING.get(wrap_selected_cutout, [])], model_image,
                    CUTOUT_MAPPING.get(wrap_selected_cutout, []))
                st.image(model_image, use_column_width=True)

    repaint_cloth = st.button("Repaint Cloth")

    if repaint_cloth:
        remove_files_out_image()

    if len(cutout_files) > 0 and st.session_state.key_points_model is not None:
        col1, col2 = st.columns(2)

        with col1:

            controls_models = st.columns(3)
            default_value = "Select"

            with controls_models[0]:
                repaint_blend_intensity = st.select_slider("Intensity", range(10, 100, 10),
                                                           key="repaint_blend_intensity")

            with controls_models[1]:

                repaint_selected_cutout = st.selectbox("cutouts", CUTOUT_RANGE, index=0, key="repaint_cutouts_dropdown")
                revised_points = st.button(f"Revise pts for {repaint_selected_cutout}")
                if revised_points:
                    st.session_state.cutout_list[repaint_selected_cutout] = st.session_state.key_points_model[
                        CUTOUT_MAPPING.get(repaint_selected_cutout, [])]


            with controls_models[2]:

                repaint_selected_point_del = st.selectbox("Delete any keypoint",
                                                          [default_value] + list(range(0, len(
                                                              st.session_state.cutout_list[repaint_selected_cutout]))),
                                                          index=0, key="remove_cutout_keypoint")

                if repaint_selected_point_del != default_value:
                    st.write(f"Delete: {repaint_selected_point_del}")

                    st.session_state.cutout_list[repaint_selected_cutout] = np.delete(
                        st.session_state.cutout_list[repaint_selected_cutout], repaint_selected_point_del, axis=0)
                    st.rerun()

                add_point_post = st.selectbox("Add point after",
                                              [default_value] + list(
                                                  range(0, len(st.session_state.cutout_list[repaint_selected_cutout]))),
                                              index=min(st.session_state.last_selected_cutout_pt + 1,
                                                        len(st.session_state.cutout_list[repaint_selected_cutout])),
                                              key="add_cutout_keypoint")

            repaint_cutout = [file for file in cutout_files if repaint_selected_cutout in file]
            repaint_cutout_image = np.asarray(
                Image.open(os.path.join('myra-app-main/predict/images/cutouts', repaint_cutout[0])))

            repaint_cutout_image_processed = process_cutout(repaint_cutout_image, np.asarray(model_image))

            # Define the weight for each image
            alpha = repaint_blend_intensity / 100  # Weight for the model image
            beta = (100 - repaint_blend_intensity) / 100  # Weight for the cutout image

            repaint_blended_image = cv2.addWeighted(np.asarray(model_image)[:, :, :3], alpha,
                                                    repaint_cutout_image_processed, beta, 0)
            repaint_blended_image = Image.fromarray(repaint_blended_image.astype(np.uint8))

            write_points_and_labels_over_image(st.session_state.cutout_list[repaint_selected_cutout],
                                               repaint_blended_image, None)

            if os.path.exists('myra-app-main/predict/images/out_image.png'):

                out_image_arr = np.asarray(Image.open('myra-app-main/predict/images/out_image.png')).copy()
                out_mask_arr = np.asarray(Image.open('myra-app-main/predict/images/out_mask.png')).copy()


            else:

                out_image_arr = np.array(Image.open(IMAGE_ADDRESS)).copy()
                ag_mask_arr = np.asarray(Image.open(AG_MASK_ADDRESS))
                mask_image_arr = np.asarray(Image.open(PARSE_ADDRESS))
                model_image_arr = np.asarray(Image.open(IMAGE_ADDRESS)).copy()

                out_image_arr, out_mask_arr = initialize_out_image_and_mask(out_image_arr,
                                                                            model_image_arr, mask_image_arr,
                                                                            ag_mask_arr)
                skin_mask = np.asarray(Image.open(SKIN_MASK_ADDRESS))[:, :, 0]

                out_image_arr[skin_mask == 255, :3] = model_image_arr[skin_mask == 255, :3]

                out_mask_arr[skin_mask == 255] = 255

            l_mask = np.zeros((1024, 768))
            cutout_points_list = [st.session_state.cutout_list[repaint_selected_cutout].astype(np.int32)]
            cv2.fillPoly(l_mask, cutout_points_list, 255)
            l_mask = l_mask.astype(np.uint32)

            out_image_arr[l_mask == 255, :3] = 0
            out_mask_arr[l_mask == 255] = 0
            if not os.path.exists('myra-app-main/predict/images/out_image_l_mask_ag.png'):
                Image.fromarray(out_image_arr.astype(np.uint8)).save(
                    'myra-app-main/predict/images/out_image_l_mask_ag.png')
            if not os.path.exists('myra-app-main/predict/images/out_mask_l_mask_ag.png'):
                Image.fromarray(out_mask_arr.astype(np.uint8)).save(
                    'myra-app-main/predict/images/out_mask_l_mask_ag.png')

            ### Add new keypoint

            point_added = None
            if add_point_post != default_value:
                point_added = streamlit_image_coordinates(
                    repaint_blended_image,
                    key="cutout_pt_add")

        with col2:

            if point_added and (point_added["x"] != st.session_state.point_selected_cutout["x"] and point_added["y"] !=
                                st.session_state.point_selected_cutout["y"]):
                st.session_state.point_selected_cutout = point_added
                value_to_append = np.array([point_added["x"], point_added["y"]], dtype=np.float32)

                # st.session_state.cutout_points = np.append(st.session_state.cutout_points, [value_to_append], axis=0)
                st.session_state.last_selected_cutout_pt = add_point_post + 1
                st.session_state.cutout_list[repaint_selected_cutout] = np.insert(
                    st.session_state.cutout_list[repaint_selected_cutout],
                    add_point_post + 1,
                    [value_to_append], axis=0)

                st.rerun()

            if os.path.exists('myra-app-main/predict/images/out_image_l_mask_ag.png'):
                out_image = np.asarray(Image.open('myra-app-main/predict/images/out_image_l_mask_ag.png')).copy()
            if os.path.exists('myra-app-main/predict/images/out_mask_l_mask_ag.png'):
                out_mask = np.asarray(Image.open('myra-app-main/predict/images/out_mask_l_mask_ag.png')).copy()

            out_image[l_mask == 255, :3] = repaint_cutout_image[l_mask == 255, :3]

            out_mask[l_mask == 255] = 255
            modify = st.button(f"Save Modification")

            if modify:
                model_image = np.asarray(Image.open(IMAGE_ADDRESS)).copy()
                mask_image = np.asarray(Image.open(PARSE_ADDRESS))

                # out_image[mask_image == 11, :] = model_image[mask_image == 11, :]

                #if repaint_selected_cutout == 'backbone':

                   # out_image[mask_image == 5, :] = model_image[mask_image == 5, :]
                   # out_image[mask_image == 6, :] = model_image[mask_image == 6, :]
                Image.fromarray(out_image.astype(np.uint8)).save('myra-app-main/predict/images/out_image.png')
                Image.fromarray(out_mask.astype(np.uint8)).save('myra-app-main/predict/images/out_mask.png')
                os.remove('myra-app-main/predict/images/out_image_l_mask_ag.png')
                os.remove('myra-app-main/predict/images/out_mask_l_mask_ag.png')

            st.image(out_image.astype(np.uint8))

    ###########Replant Skin######################

    replant_skin = st.button("Replant Masks")

    if replant_skin:
        st.session_state.select_colour = False
        st.session_state.skin_out_image_toggle = False
        if (os.path.exists('myra-app-main/predict/images/skin_image.png')):
            os.remove('myra-app-main/predict/images/skin_image.png')
        if (os.path.exists('myra-app-main/predict/images/replant_image.png')):
            os.remove('myra-app-main/predict/images/replant_image.png')
        st.session_state.skin_replant_image_arr = None
        st.session_state.skin_mask_replant_image_arr = None

    # Display a file uploader widget
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    controls_models = st.columns(5)

    with controls_models[0]:
        skin_blend_intensity = st.select_slider("Intensity", range(0, 100, 10), key="skin_blend_intensity")
        default_value = "Select"

        skin_mask_selected_point_del = st.selectbox("Delete any keypoint",
                                                    [default_value] + list(range(0, len(
                                                        st.session_state.skin_mask_selected_points) + 2)),
                                                    index= None, key="remove_skin_mask_keypoint")


        if skin_mask_selected_point_del is not None :
            st.write(f"Delete: {skin_mask_selected_point_del}")

            st.session_state.skin_mask_selected_points = np.delete(
                st.session_state.skin_mask_selected_points, skin_mask_selected_point_del, axis=0)



        add_skin_point_index = st.selectbox("Add point at Index",
                                            [default_value] + list(
                                                range(0, 100)),
                                            index=None,
                                            key="add_skin_mask_keypoint")

        if add_skin_point_index is not None:
            st.write(f"YOU HAVE PRESELECTED AN INDEX {add_skin_point_index}")


    with controls_models[1]:
        select_colour = st.button("Select Colour")
        if select_colour:
            st.session_state.select_colour = True
            st.session_state.skin_mask_replace_color = None


        back_image = st.selectbox("Back image", BACK_IMAGES, index=0, key="back_image")
        skin_image = st.selectbox("Front Image", FRONT_IMAGES, index=1, key="front_image")

        if back_image != "uploaded_image":
            try:
                back_image = Image.open(IMAGES_MAPPING[back_image])
                back_image_arr = np.array(back_image)
            except Exception as e:
                st.write(str(e))
        else:
            if uploaded_file is None:
                st.write("PLEASE SELECT A FILE FIRST")
            back_image = Image.open(uploaded_file)
            expected_shape = (1024, 768, 3)
            back_image_arr = np.array(back_image)
            if back_image_arr.shape != expected_shape:
                back_image = back_image.resize((expected_shape[1], expected_shape[0]))
                back_image_arr = np.asarrayback_image()


        back_image_arr = convert_to_rgb(back_image_arr, COLORMAP)


        if skin_image != "uploaded_image":
            try:
                skin_image = Image.open(IMAGES_MAPPING[skin_image])
                skin_image_arr = np.asarray(skin_image)
            except Exception as e:
                st.write(str(e))


        else:

            if uploaded_file is None:
                st.write("PLEASE SELECT A FILE FIRST")
            skin_image = Image.open(uploaded_file)

            expected_shape = (1024, 768, 3)
            skin_image_arr = np.array(skin_image)

            if skin_image_arr.shape != expected_shape:

                skin_image = skin_image.resize((expected_shape[1], expected_shape[0]))
                skin_image_arr = np.asarray(skin_image)





        skin_image_arr = convert_to_rgb(skin_image_arr, COLORMAP)

        if st.session_state.skin_replant_image_arr is None:
            st.session_state.skin_replant_image_arr = skin_image_arr





    with controls_models[2]:

        replant_colour = st.button("Replant with colour")
        if replant_colour:

            replant_image_arr = st.session_state.skin_replant_image_arr
            if st.session_state.skin_mask_replace_color is not None:
                color = replant_image_arr[
                    st.session_state.skin_mask_replace_color["y"], st.session_state.skin_mask_replace_color["x"]]
            else:
                color = np.array([0, 0, 0])
            l_mask = np.zeros((1024, 768))

            skin_mask_cutout_points_list = [np.array(st.session_state.skin_mask_selected_points).astype(np.int32)]


            cv2.fillPoly(l_mask, skin_mask_cutout_points_list, 255)

            if len(replant_image_arr.shape) > 2:

                l_mask = l_mask.astype(np.uint8)
                replant_image_arr = replant_image_arr.astype(np.uint8)
                replant_image_arr[l_mask == 255, :3] = color[:3]



            else:
                replant_image_arr[l_mask == 255] = color[0]

            st.write("hi")
            if os.path.exists("myra-app-main/predict/images/out_mask.png"):
                if st.session_state.skin_mask_replant_image_arr is None:
                    mask_replant_image = "myra-app-main/predict/images/out_mask.png"
                    mask_replant_image = Image.open(mask_replant_image)
                    st.session_state.skin_mask_replant_image_arr = np.array(mask_replant_image).copy()
                st.session_state.skin_mask_replant_image_arr[l_mask == 255] = 255


            if np.all(color == 0):
                replant_image_arr[l_mask == 255] = 0


            st.session_state.skin_replant_image_arr = replant_image_arr

            st.session_state.select_colour = False

    with controls_models[3]:

        clear  = st.button("clear Mask")
        if clear:
            st.session_state.skin_mask_selected_points = []
            st.session_state.select_colour = False



        save_and_clear = st.button("Save and Clear Mask")

        if save_and_clear:
            st.session_state.skin_mask_selected_points_history.insert(0, st.session_state.skin_mask_selected_points)
            st.session_state.skin_mask_selected_points = []
            st.session_state.select_colour = False


        if len(st.session_state.skin_mask_selected_points_history) > 0:
            default_value = "defaut"

            select_mask = st.selectbox("Select Mask",
                                                  [default_value] + list(range(0,
                                                      len(st.session_state.skin_mask_selected_points_history))),
                                                  index=0, key="select_mask")

            if select_mask != default_value and  st.session_state.selected_skin_mask != select_mask:
                st.session_state.selected_skin_mask = select_mask


                st.session_state.skin_mask_selected_points = st.session_state.skin_mask_selected_points_history[select_mask]
                #st.rerun()







    with controls_models[4]:

        replant_skin = st.button("Replant with skin")

        if replant_skin :

            if not st.session_state.skin_out_image_toggle:

                replant_image = "myra-app-main/predict/images/out_image.png"
                replant_image = Image.open(replant_image)
                st.session_state.skin_replant_image_arr = np.array(replant_image).copy()
                st.session_state.skin_out_image_toggle = True

                mask_replant_image = "myra-app-main/predict/images/out_mask.png"
                mask_replant_image = Image.open(mask_replant_image)
                st.session_state.skin_mask_replant_image_arr = np.array(mask_replant_image).copy()




            l_mask = np.zeros((1024, 768))

            skin_mask_cutout_points_list = [np.array(st.session_state.skin_mask_selected_points).astype(np.int32)]
            cv2.fillPoly(l_mask, skin_mask_cutout_points_list, 255)
            l_mask = l_mask.astype(np.uint8)
            skin_image_arr = skin_image_arr.astype(np.uint8)
            st.session_state.skin_replant_image_arr = st.session_state.skin_replant_image_arr.astype(np.uint8)
            st.session_state.skin_replant_image_arr[l_mask == 255, :3] = skin_image_arr[l_mask == 255, :3]

            st.session_state.skin_mask_replant_image_arr[l_mask == 255] = 255



    col1, col2 = st.columns(2)
    with col1:


        if skin_image is not None:

            # Define the weight for each image
            alpha = skin_blend_intensity / 100  # Weight for the model image
            beta = (100 - skin_blend_intensity) / 100  # Weight for the cutout image


            ## To Add Transparency
            #skin_mask = np.asarray(Image.open(SKIN_MASK_ADDRESS))
            #back_image_arr[skin_mask[:,:,0] == 0] = skin_image_arr[skin_mask[:,:,0] == 0]

            skin_blended_image = cv2.addWeighted(back_image_arr[:, :, :3], alpha,
                                                 skin_image_arr[:, :, :3].astype(np.uint8), beta, 0)

            skin_blended_image = Image.fromarray(skin_blended_image.astype(np.uint8))

            write_points_and_labels_over_image(
                np.array(st.session_state.skin_mask_selected_points), skin_blended_image)

            edit_point = streamlit_image_coordinates(
                skin_blended_image,
                key="skin_blended_image")

            if edit_point and (edit_point["x"] != st.session_state.point_selected_skin_mask["x"] and edit_point["y"] !=
                               st.session_state.point_selected_skin_mask["y"]):


                st.session_state.point_selected_skin_mask = edit_point
                st.write(st.session_state.select_colour)
                if st.session_state.select_colour:
                    st.session_state.skin_mask_replace_color = edit_point
                else:

                    try:
                        value_to_append = np.array([edit_point["x"], edit_point["y"]], dtype=np.float32)

                        if len(st.session_state.skin_mask_selected_points) == 0:
                            st.session_state.skin_mask_selected_points = np.empty((0, 2))

                        if add_skin_point_index  is None:

                            st.session_state.skin_mask_selected_points = \
                                np.vstack([st.session_state.skin_mask_selected_points, value_to_append])
                        else:


                            st.session_state.skin_mask_selected_points = np.insert(
                                st.session_state.skin_mask_selected_points,
                                add_skin_point_index,
                                [value_to_append], axis=0)



                        write_points_and_labels_over_image(
                        np.array(st.session_state.skin_mask_selected_points), skin_blended_image)
                        st.rerun()
                    except Exception as e:

                        st.write(str(e))

        with col2:

            save = st.button("save")




            if save:

                if len(np.asarray(skin_image).shape) == 2:

                    # Create a reverse colormap from RGB to label
                    reverse_colormap = {tuple(v): k for k, v in COLORMAP.items()}

                    # Create an empty labeled image array
                    labeled_image_array = np.zeros((st.session_state.skin_replant_image_arr.shape[0],
                                                    st.session_state.skin_replant_image_arr.shape[1]), dtype=np.uint8)

                    # Map the RGB image back to labels
                    for rgb_color, label in reverse_colormap.items():

                        mask = np.all(st.session_state.skin_replant_image_arr == rgb_color, axis=-1)
                        labeled_image_array[mask] = label

                    # Convert the array back to an image

                    labeled_image = Image.fromarray(labeled_image_array)

                    labeled_image.save('myra-app-main/predict/images/replant_image.png')
                else:

                    Image.fromarray(st.session_state.skin_replant_image_arr).save(
                        'myra-app-main/predict/images/replant_image.png')
                    if st.session_state.skin_mask_replant_image_arr is not None:
                        Image.fromarray(st.session_state.skin_mask_replant_image_arr).save(
                            'myra-app-main/predict/images/replant_mask_image.png')

            st.image(Image.fromarray(st.session_state.skin_replant_image_arr.astype(np.uint8)))

    # Create a button

    button_clicked = st.button("Run Diffusion!")

    image_str = generate_random_string(12)
    cloth_str = generate_random_string(12)
    ag_mask_str = generate_random_string(12)
    skin_mask_str = generate_random_string(12)
    parse_str = generate_random_string(12)
    parse_ag_str = generate_random_string(12)
    out_image_str = generate_random_string(12)
    out_mask_str = generate_random_string(12)

    if button_clicked:
        st.write("Button clicked!")
        st.write(IMAGE_ADDRESS)
        image = cloudinary_upload.uploadImage(IMAGE_ADDRESS, image_str)
        st.write("Button clicked!")
        cloth = cloudinary_upload.uploadImage(CLOTH_ADDRESS, cloth_str)
        ag_mask = cloudinary_upload.uploadImage(AG_MASK_ADDRESS, ag_mask_str)
        skin_mask = cloudinary_upload.uploadImage(SKIN_MASK_ADDRESS, skin_mask_str)
        parse = cloudinary_upload.uploadImage(PARSE_ADDRESS, parse_str)
        parse_ag = cloudinary_upload.uploadImage(PARSE_AG_ADDRESS, parse_ag_str)
        st.write("Button clicked!")
        out_image = cloudinary_upload.uploadImage('myra-app-main/predict/images/out_image.png', out_image_str)
        out_mask = cloudinary_upload.uploadImage('myra-app-main/predict/images/out_mask.png', out_mask_str)
        st.write("Button clicked!")
        output = replicate.run(
            "shrikantbhole/myra:6baecf1816d4ef23da536f7a59cafef418ce04edad10bbac6c07161950e82f1d",
            input={
                "image": image,
                "cloth": cloth,
                "ag_mask": ag_mask,
                "skin_mask": skin_mask,
                "parse": parse,
                "parse_ag": parse_ag,
                "s_pos": get_s_pos_string(s_pos_json),
                "c_pos": get_c_pos_string(c_pos_json),
                "out_image": out_image,
                "out_mask": out_mask
            }
        )
        st.write("Button clicked!")

        print(output)
        response = requests.get(output)
        final_image = np.array(Image.open(BytesIO(response.content)))

        st.image(final_image, caption='final Image')



def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates home based on these inputs.
    """
    # submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt = configure_sidebar()
    try:
        main_page()
    except Exception as e:
        1 == 1


def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application,
    including the form for user inputs and the resources section.
    """
    with st.sidebar:
        home = st.button(":rainbow[**Our Creative**]")
        if home:
            st.session_state.page = "home"
        with st.expander(":rainbow[**Try Yourself**]"):


            st.session_state.token = st.text_input(':rainbow[**Please enter token**] ')
            enter = st.button("enter")
            if enter and st.session_state.token == "myra2024":
                st.write("correct")
                st.session_state.page = "main"



        with st.expander(":rainbow[**Setup Demo or Request Token 👋🏾 💌 🚀**]"):
            email = st.text_input('email', key='demo_email')
            contact = st.text_input('contact (Optional)', key='demo_contact')
            body = st.text_area('Body', key='demo_body')

            if st.button("Send Email", key = 'demo_send'):
                try:
                    if email == '':
                        st.sidebar.write("Please enter email id")
                    else:
                        msg = MIMEText(body)
                        msg['From'] = 'bholeshrikant@gmail.com'
                        msg['To'] = 'bholeshrikant@gmail.com'
                        msg['Subject'] = email + "---" + contact

                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login('bholeshrikant@gmail.com', 'kybp iedf qlba zdmn')
                        server.sendmail('bholeshrikant@gmail.com', 'bholeshrikant@gmail.com', msg.as_string())
                        server.quit()

                        st.success('Email sent successfully! 🚀')
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

        with st.expander(":rainbow[**Reach out on Watsapp**]"):
            st.write(":orange[ Just drop us message on 8800283739 and we will reach out you soon")

        with st.expander(":rainbow[A bit about us👋🏾]"):
            st.write(":orange[ Nothing fam !! Just some bunch of tech enthusiasts working persistently to craft  some Myra Magic into fashion Industry  ]")

            st.markdown(
            """   
            
            Website -> [Flexli](https://flexli.in/)
            
            LinkedIn → [Shrikant](https://www.linkedin.com/in/tonykipkemboi)
            
            """
        )

        # return submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt
        return;


if __name__ == "__main__":
    configure_sidebar()
    if st.session_state.page == "main":
        main()
    if st.session_state.page == "home":
        home_page()
