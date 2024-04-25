import streamlit as st
from streamlit_image_select import image_select
from PIL import Image, ImageDraw, ImageFont
import replicate
import requests
from io import BytesIO
from upload_images import cloudinary_upload
import os
import torch
from predict import predict_pos_keypoints as kg
from predict import predict_parse_seg_image as pg
from roboflow_apis import  fetch_model_segmentation_image as rb
import numpy as np
import json
import cv2
from torchvision import transforms
from streamlit_image_coordinates import streamlit_image_coordinates
from predict_tps import  generate_tps, generate_tps_1
from datasets.dataset_st import  get_s_pos, get_c_pos

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

## Uploaded image location
TSHIRT_IMAGE_ADDRESS = "myra-app-main/upload_images/tshirt.jpg"
MODEL_IMAGE_ADDRESS = "myra-app-main/upload_images/image.jpg"
MODEL_SEG_IMAGE_ADDRESS = "myra-app-main/predict/images/parse.png"
AG_MASK_ADDRESS = "myra-app-main/predict/images/ag_mask.png"
SKIN_MASK_ADDRESS = "myra-app-main/predict/images/skin_mask.png"

if os.path.exists('myra-app-main/out_image.jpg'):
    os.remove('myra-app-main/out_image.jpg')
if os.path.exists('myra-app-main/out_mask.jpg'):
    os.remove('myra-app-main/out_mask.jpg')
if os.path.exists('myra-app-main/pg_output.png'):
    os.remove('myra-app-main/pg_output.png')


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
if 'point_selected' not in st.session_state:
    st.session_state.point_selected = {"x": 0, "y": 0}



# We will be overwriting on tshirt image highlighting keypoints circles with their  labels
# We need to store in session history cover area boundaries of all key points circles and labels, since when keypoint in modified original img crop could be  brought back
def write_cover_areas_for_pointer_and_labels(arr: np.ndarray, image: Image, cover_area_pointer_list: list,
                                             cover_area_label_list: list):

    point_size = 5
    label_text = 'Point'

    draw = ImageDraw.Draw(image)
    font_size = 16
    font = ImageFont.truetype("arial.ttf", font_size)

    text_width, text_height = 5,5



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
    text_width, text_height = 5,5
    point_color = 'green'

    for id, point in enumerate(arr):
        draw.ellipse(
            (point[0] - point_size, point[1] - point_size, point[0] + point_size, point[1] + point_size),
            fill=point_color)
        text_x = point[0] + point_size + 5  # Adjust for spacing
        text_y = point[1] - text_height // 2
        draw.text((text_x, text_y), str(id), fill='green', font=font)
    return image


def update_point_over_image(edited_image: Image, node: str, value: dict, kp_arr: np.ndarray,
                            cover_area_pointer_list: list, cover_area_label_list: list):
    point_size = 5
    label_text = 'Point'
    draw = ImageDraw.Draw(edited_image)
    font_size = 16
    font = ImageFont.truetype("arial.ttf", font_size)
    text_width, text_height = 5,5

    node = int(node)

    kp_arr[node][0] = value["x"]
    kp_arr[node][1] = value["y"]

    original_image = Image.open(TSHIRT_IMAGE_ADDRESS)
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


def main_page() -> None:
    """Main page layout and logic for generating myra_v1."""

    #############PLOT KEYPOINTS OVER TSHIRT#############
    # SAVE UPLOADED TSHIRT IMAGE AND MODEL IMAGE
    f1 = st.file_uploader("Please choose tshirt Image")
    f2 = st.file_uploader("Please choose Model Pos image ")

    show_file = st.empty()
    if not f1:
        show_file.info("Please upload an image")

    if isinstance(f1, BytesIO):
        image = Image.open(f1)
        image.save('myra-app-main/upload_images/tshirt.jpg')

    if not f2:
        show_file.info("Please upload an image")

    if isinstance(f2, BytesIO):
        image = Image.open(f2)
        image.save('myra-app-main/upload_images/image.jpg')

    # Create two columns to show cloth and model Image
    col1, col2 = st.columns(2)

    # Display the myra_v1 in the column along with keypoints

    with col1:
        # Create an input text box to select a keypoint whose position needs to be changed

        node = st.text_input('Enter node position to change')
        if node:
            st.write("You are modifying Node " + node + "   Please click on new position")


        if os.path.exists(TSHIRT_IMAGE_ADDRESS):


            with open('myra-app-main/data/00006_00/cloth_landmark_json.json', 'r') as file:

                json_list = json.load(file)
                kp_arr = np.array(json_list["long"]) * 250
                if st.session_state.key_points_tshirt is None:
                    st.session_state.key_points_tshirt = kp_arr

            image = Image.open(TSHIRT_IMAGE_ADDRESS)

            if not st.session_state.cover_area_pointer_list_tshirt:

                write_cover_areas_for_pointer_and_labels(kp_arr, image, st.session_state.cover_area_pointer_list_tshirt,
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
                                            st.session_state.cover_area_label_list_tshirt)


                else:
                    st.sidebar.write("Please select a node first")

                # out_image = cloudinary_upload.uploadImage('myra-app-main/upload_images/tshirt.jpg', 'tshirt')

    with col2:
        image = Image.open(TSHIRT_IMAGE_ADDRESS)
        write_points_and_labels_over_image(st.session_state.key_points_tshirt, image)
        st.image(image, use_column_width=True)

        ###################KEYPOINT DETECTOR#########################
    # Create two columns to show cloth and model Image
    col1, col2 = st.columns(2)

    # Display the myra_v1 in the columns

    with col1:


        model_image = Image.open(MODEL_IMAGE_ADDRESS)
        if st.session_state.key_points_model is not None:
            write_points_and_labels_over_image(st.session_state.key_points_model, model_image)

        # If Key Point Detector Is called
        key_point_detector = st.button("Run KeyPoint Detector!")
        if key_point_detector:
            key_points = st.session_state.key_points_tshirt

            p_pos = kg.get_p_pos(key_points)

            st.session_state.key_points_model = p_pos

            model_image = Image.open(MODEL_IMAGE_ADDRESS)
            write_points_and_labels_over_image(p_pos, model_image)
            st.session_state.cover_area_pointer_list_model = []
            st.session_state.cover_area_label_list_model = []
            write_cover_areas_for_pointer_and_labels(p_pos, model_image,
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
                update_point_over_image(model_image, model_node, model_value, st.session_state.key_points_model,
                                        st.session_state.cover_area_pointer_list_model,
                                        st.session_state.cover_area_label_list_model)

            else:
                st.sidebar.write("Please select a node first")

    with col2:
        model_image = Image.open(MODEL_IMAGE_ADDRESS)
        write_points_and_labels_over_image(st.session_state.key_points_model, model_image)
        st.image(model_image, use_column_width=True)


    ###########PARSE GENERATOR PIPELINE######################
    # Create two columns to show cloth and model Image
    col1, col2 = st.columns(2)

    with col1:
        model_parse_generator = st.button("Run Model Parse Generator!")
        if model_parse_generator:
            p_pos = st.session_state.key_points_model.copy()
            p_pos = torch.tensor(p_pos)
            p_pos[:, 0] = p_pos[:, 0] / 768
            p_pos[:, 1] = p_pos[:, 1] / 1024
            p_pos = p_pos.float()

            #model_seg_image, ag_mask, tshirt_mask = rb.predict_seg_img(MODEL_IMAGE_ADDRESS)
            model_seg_image = Image.open("myra-app-main/data/00006_00/parse.png")

            pg_output,parse13_model_seg = pg.parse_model_seg_image(get_s_pos(), p_pos, model_seg_image)

            ag_mask = 255 - cv2.imread(AG_MASK_ADDRESS, cv2.IMREAD_GRAYSCALE)
            ag_mask  = np.array(transforms.Resize(768)(Image.fromarray(ag_mask)))
            skin_mask = cv2.imread(SKIN_MASK_ADDRESS, cv2.IMREAD_GRAYSCALE)
            skin_mask =  np.array(transforms.Resize(768)(Image.fromarray(skin_mask)))
            parse_seg_image = pg.draw_parse_model_image(pg_output)
            st.image(parse_seg_image)
            c_pos, v_pos = get_c_pos()

            p_pos = st.session_state.key_points_model.copy()
            p_pos = torch.tensor(p_pos)
            p_pos[:, 0] = p_pos[:, 0] / 768
            p_pos[:, 1] = p_pos[:, 1] / 1024
            out_image, out_mask = generate_tps_1(
                np.asarray(Image.open(MODEL_IMAGE_ADDRESS)),
                np.asarray(Image.open(TSHIRT_IMAGE_ADDRESS)),
                v_pos.float(),
                torch.tensor(p_pos.float()),
                ag_mask,
                skin_mask,
                parse13_model_seg.squeeze()

            )
            st.image(Image.fromarray(out_image))
            st.image(Image.fromarray(out_mask))

    with col2:
        print("Hi")


    #############DIFFUSION INFERENCE PIPELINE#########################

    f3 = st.file_uploader("Please choose Warped Output Image")
    f4 = st.file_uploader("Please choose Warped Mask Image")
    f5 = st.file_uploader("Please choose Segmentation output Image")

    if not f3:
        show_file.info("Please upload an image")

    if isinstance(f3, BytesIO):
        image = Image.open(f3)
        image.save('myra-app-main/upload_images/out_image.png')

    if not f4:
        show_file.info("Please upload an image")

    if isinstance(f4, BytesIO):
        image = Image.open(f4)
        image.save('myra-app-main/upload_images/out_mask.png')

    if not f5:
        show_file.info("Please upload an image")

    if isinstance(f5, BytesIO):
        image = Image.open(f5)
        image.save('myra-app-main/upload_images/pg_output.png')
        # Create two columns for displaying myra_v1 side by side
        col1, col2, col3 = st.columns(3)

    # Display the myra_v1 in the columns
    with col1:
        if (os.path.exists('myra-app-main/upload_images/out_image.jpg')):
            st.image('myra-app-main/upload_images/out_image.jpg', caption='Output Image', use_column_width=True)
            out_image = cloudinary_upload.uploadImage('myra-app-main/upload_images/out_image.jpg', 'out_image')

    with col2:
        if (os.path.exists('myra-app-main/uploaded_images/out_mask.jpg')):
            st.image('myra-app-main/upload_images/out_mask.jpg', caption='Mask Image', use_column_width=True)
            out_mask = cloudinary_upload.uploadImage('myra-app-main/upload_images/out_mask.jpg', 'out_mask')

    with col3:
        if (os.path.exists('myra-app-main/uploaded_images/pg_output.png')):
            st.image('myra-app-main/upload_images/pg_output.png', caption='PG Output Image', use_column_width=True)
            pg_output = cloudinary_upload.uploadImage('myra-app-main/upload_images/pg_output.png', 'pg_output')

    # Create a button
    button_clicked = st.button("Run Diffusion!")

    # Check if the button is clicked
    if button_clicked:
        st.write("Button clicked!")
        output = replicate.run(
            "shrikantbhole/diffusion:de6b9511fa9e4e22c8ae85e090ba5f1343e73e0ff581fdb19481dd1e12837ba0",
            input={
                "out_mask_file": out_mask,
                "out_image_file": out_image,
                "pg_output_file": pg_output,
                "out_image_file_2": "https://replicate.delivery/pbxt/KdXQXAymtIImLDUjy8JPaMcrUuMoSAZJukGlA46JFW0RFHgp/_0_207.jpeg"
            }
        )

        print(output)
        response = requests.get(output)
        final_image = np.array(Image.open(BytesIO(response.content)))
        st.image(final_image, caption='final Image', use_column_width=True)

        st.write(
            "<span style='font-family: Roboto, sans-serif;'>At Myra, our vision is to inject a sense of magic and creativity into e-commerce fashion "
            "photoshoots by leveraging AI-generated models. </span>",
            unsafe_allow_html=True)
        st.write(" ")

        st.write(
            "<span style='font-family: Roboto, sans-serif;'> Here's the concept: We begin with an image of a mannequin showcasing the fashion product under ideal "
            "lighting conditions. This image, along with your prompt detailing the desired attributes of the final model, is fed into our Myra AI system. "
            "From there, Myra AI swiftly crafts the perfect image of real looking model tailored to your specifications in no time.</span>",
            unsafe_allow_html=True)

        st.write(
            "<span style='font-family: Roboto,sans-serif'>We have listed below few examples of results obtained from Myra AI .The mannequin myra_v1 below were retrieved from the internet from different sites. "
            "pinterest, shutterstock, istockphoto. "
            " Myra AI could "
            "fit AI models within this dress, maintaining the dress outline, tone, and appearance.</span>",
            unsafe_allow_html=True)

        st.write("")
        M1 = image_select(
            label="WHITE CREAM COLOUR FULL TSHIRT ON BLUE JEANS (https://www.pinterest.com/pin/30891947430775019)",
            images=[
                "myra-app-main/myra_v1/M1/input.png", "myra-app-main/myra_v1/M1/2-2.png",
                "myra-app-main/myra_v1/M1/2-2.png", "myra-app-main/myra_v1/M1/3-5.png", "myra-app-main/myra_v1/M1/4-3.png",
                "myra-app-main/myra_v1/M1/5-3.png"

            ],
            captions=["Input Image to Myra AI fetched  from pinterest website",
                      "Charismatic Young Man, Radiating Confidence and Charm, Captured in Stunning 8K Resolution.",
                      "Smart, Handsome Male Model,Bold Beard, His Eyes Piercing with Clarity, Embracing the Cinematic Aura ",
                      "Striking Portrait of a Dynamic and Attractive Male Model, Bold Expressions , Canon's 8K Resolution",
                      "A cute spotless perfect looking model, Unleash the Beauty, Cinematic Touch of Canon Photography",
                      "Create a Masterpiece: Generate an Image of a Smart and Dynamic Male Model, His Expressions Bold",
                      ],
            use_container_width=False
        )
        st.write(
            "<span style='font-family: Roboto,sans-serif'>PLEASE CLICK ON ANY ABOVE IMAGE FOR A CLOSER LOOK.</span>",
            unsafe_allow_html=True)
        if M1 != "myra_v1/M1/input.png":
            # Load the myra_v1
            image1 = Image.open("myra-app-main/myra_v1/M1/input.png")
            image2 = Image.open(M1)

            # Create two columns for displaying myra_v1 side by side
            col1, col2 = st.columns(2)

            # Display the myra_v1 in the columns
            with col1:
                st.image(image1, caption='Input Image', use_column_width=True)

            with col2:
                st.image(image2, caption='Output Image', use_column_width=True)

        st.write("")
        st.write("")
        st.write("")
        M6 = image_select(
            label="RED COLLAR DOTTED TSHIRT ON BLUE JEANS (https://www.shutterstock.com/image-photo/fulllength-male-mannequin-dressed-tshirt-jeans-1067987750)",
            images=[
                "myra-app-main/myra_v1/M6/input.png", "myra-app-main/myra_v1/M6/1-4.png",
                "myra-app-main/myra_v1/M6/2.png", "myra-app-main/myra_v1/M6/3-5.png", "myra-app-main/myra_v1/M6/4-3.png",
                "myra-app-main/myra_v1/M6/5-2.png"

            ],
            captions=["Input image of Mannequin with tshirt obtained from shutterstock website",
                      "Attractive Young Man, His Eyes Reflecting Clarity and Confidence, Transformed into Cinematic Splendor ",
                      "An artistic model, more idealistic and perfectionist, AI magic, Enshrined in the Timeless Beauty ",
                      "A real looking smiling model, natural model photoshoots, Radiating Youthful Energy and Charm, 8k",
                      "Dynamic and Attractive Male Model, His Bold Expressions and Charismatic Presence Transcending the Screen,",
                      "Smart and Handsome Male Model, Bold beard, picture perfect, His Eyes Sparkling with Clarity and Intelligence",
                      ],
            use_container_width=False
        )
        st.write(
            "<span style='font-family: Roboto,sans-serif'>PLEASE CLICK ON ANY ABOVE IMAGE FOR A CLOSER LOOK.</span>",
            unsafe_allow_html=True)
        if M6 != "myra_v1/M6/input.png":
            # Load the myra_v1
            image1 = Image.open("myra-app-main/myra_v1/M6/input.png")
            image2 = Image.open(M6)

            # Create two columns for displaying myra_v1 side by side
            col1, col2 = st.columns(2)

            # Display the myra_v1 in the columns
            with col1:
                st.image(image1, caption='Input Image', use_column_width=True)

            with col2:
                st.image(image2, caption='Output Image', use_column_width=True)

        st.write("")
        st.write("")
        st.write("")
        M7 = image_select(
            label="White half sleeves tshirt on blue jeans (https://www.istockphoto.com/photo/full-length-male-mannequin-gm1289535860-385180516)",
            images=[
                "myra-app-main/myra_v1/M7/input.png", "myra-app-main/myra_v1/M7/1-2.png",
                "myra-app-main/myra_v1/M7/2-2.png", "myra-app-main/myra_v1/M7/3-2.png", "myra-app-main/myra_v1/M7/4-3.png",

            ],
            captions=["Input image of Mannequin with tshirt obtained from istockphoto  website",
                      "Bring Life to the Lens, real looking, natural features,Striking Portrait of a Dynamic and Attractive Male Mode",
                      "Artistic model, AI perfect complexion and features, masterpiece,  his Eyes Sparkling with Clarity and Intelligence",
                      "Handsome bearded model, Craft an Iconic Image,Cinematic Touch of Canon Photography, Bold Expressions ",
                      "A real looking smiling model, natural model photoshoot, Aura Radiating Youthful Energy and Charm"
                      ],
            use_container_width=False
        )
        st.write(
            "<span style='font-family: Roboto,sans-serif'>PLEASE CLICK ON ANY ABOVE IMAGE FOR A CLOSER LOOK.</span>",
            unsafe_allow_html=True)
        if M7 != "myra_v1/M7/input.png":
            # Load the myra_v1
            image1 = Image.open("myra-app-main/myra_v1/M7/input.png")
            image2 = Image.open(M7)

            # Create two columns for displaying myra_v1 side by side
            col1, col2 = st.columns(2)

            # Display the myra_v1 in the columns
            with col1:
                st.image(image1, caption='Input Image', use_column_width=True)

            with col2:
                st.image(image2, caption='Output Image', use_column_width=True)

        st.write("")
        st.write("")
        st.write("")


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
