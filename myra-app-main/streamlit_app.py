import streamlit as st

## Set Page framework
st.set_page_config(page_title="Myra",
                   page_icon=":bridge_at_night:",
                   layout="wide")
st.markdown("# :rainbow[Myra - Your AI Creative Studio]")

from upload_images import cloudinary_upload

from torchvision import transforms
from math import ceil

from utility import *
from streamlit_image_coordinates import streamlit_image_coordinates
from predict.predict_tps import generate_tps_st
from predict import predict_parse_seg_image as pg
from predict import predict_pos_keypoints as kg
from dict import *
from tps_services import *
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
if 'key_points_tshirt_original' not in st.session_state:
    st.session_state.key_points_tshirt_original = None
if 'key_points_model' not in st.session_state:
    st.session_state.key_points_model = None
if 'key_points_model_original' not in st.session_state:
    st.session_state.key_points_model_original = None
if 'pg_output' not in st.session_state:
    st.session_state.pg_output = None
if 'point_selected' not in st.session_state:
    st.session_state.point_selected = {"x": 0, "y": 0}
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = '00035_00'
if 'selected_tshirt' not in st.session_state:
    st.session_state.selected_tshirt = '00034_00'


def main_page(AG_MASK_ADDRESS=None, SKIN_MASK_ADDRESS=None) -> None:
    """Main page layout and logic for generating myra_v1."""

    ######CREATE CATALOG############

    initialize('model')  ### initialize df_models sesion state
    initialize('tshirt')  ### initialize df_tshirt  sesion state
    controls_models = st.columns(3)
    files_models = st.session_state.df_models.index.tolist()

    def update(image, type):
        if type == 'model':
            st.write(f'Model: {image[:8]}')
            st.session_state.selected_model = image[:8]
            st.session_state.key_points_model_original = None
        if type == 'tshirt':
            st.write(f'Tshirt: {image[:8]}')
            st.session_state.selected_tshirt = image[:8]
            st.session_state.key_points_tshirt_original = None
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
            st.image(f'{directory_models}\{image}', caption='bike')
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
            st.image(f'{directory_tshirts}\{image}', caption='bike')
            # print(st.session_state.df_models.at[0,'selected'])
            # st.write(st.session_state.df_models.at[0,'selected'])
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

    s_pos_json = f'myra-app-main/data/openpose_json/{st.session_state.selected_model}_keypoints.json'
    c_pos_json = f'myra-app-main/data/cloth-landmark-json/{st.session_state.selected_tshirt}.json'
    IMAGE_ADDRESS = f'myra-app-main/data/image/{st.session_state.selected_model}.jpg'
    CLOTH_ADDRESS = f'myra-app-main/data/cloth/{st.session_state.selected_tshirt}.jpg'
    AG_MASK_ADDRESS = f'myra-app-main/data/ag_mask/{st.session_state.selected_model}.png'
    SKIN_MASK_ADDRESS = f'myra-app-main/data/skin_mask/{st.session_state.selected_model}.png'
    PARSE_ADDRESS = f'myra-app-main/data/parse/{st.session_state.selected_model}.png'
    PARSE_AG_ADDRESS = f'myra-app-main/data/parse_ag/{st.session_state.selected_model}.png'

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

        '''
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
        '''

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

            if st.session_state.key_points_tshirt_original is None:
                with open(c_pos_json, 'r') as file:
                    json_list = json.load(file)
                    kp_arr = np.array(json_list["long"]) * 250
                    st.session_state.key_points_tshirt = kp_arr
                    st.session_state.key_points_tshirt_original = kp_arr.copy()

            image = Image.open(CLOTH_ADDRESS)

            if not st.session_state.cover_area_pointer_list_tshirt:
                write_cover_areas_for_pointer_and_labels(st.session_state.key_points_tshirt_original, image,
                                                         st.session_state.cover_area_pointer_list_tshirt,
                                                         st.session_state.cover_area_label_list_tshirt)

            write_points_and_labels_over_image(st.session_state.key_points_tshirt_original, image)

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
                                            st.session_state.cover_area_label_list_tshirt, CLOTH_ADDRESS)


                else:
                    st.sidebar.write("Please select a node first")

                # out_image = cloudinary_upload.uploadImage('myra-app-main/upload_images/image.jpg', 'tshirt')

    with col2:
        st.write("New KeyPoints")
        image = Image.open(CLOTH_ADDRESS)
        write_points_and_labels_over_image(st.session_state.key_points_tshirt, image)
        st.image(image, use_column_width=True)

    ###################KEYPOINT DETECTOR#########################
    # Create two columns to show cloth and model Image
    col1, col2 = st.columns(2)

    # Display the myra_v1 in the columns

    with col1:

        model_image = Image.open(IMAGE_ADDRESS)
        if st.session_state.key_points_model_original is not None:
            write_points_and_labels_over_image(st.session_state.key_points_model_original, model_image)

        # If Key Point Detector Is called
        key_point_detector = st.button("Run KeyPoint Detector!")
        if key_point_detector:
            key_points = st.session_state.key_points_tshirt

            p_pos = get_p_pos(key_points, s_pos_json)

            st.session_state.key_points_model = p_pos
            st.session_state.key_points_model_original = p_pos.copy()
            model_image = Image.open(IMAGE_ADDRESS)
            write_points_and_labels_over_image(p_pos, model_image)
            st.session_state.cover_area_pointer_list_model = []
            st.session_state.cover_area_label_list_model = []
            write_cover_areas_for_pointer_and_labels(st.session_state.key_points_model_original, model_image,
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
                                        st.session_state.cover_area_label_list_model, IMAGE_ADDRESS)



            else:
                st.sidebar.write("Please select a node first")

    with col2:
        st.write("New KeyPoints")
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

        pg_output, parse13_model_seg = pg.parse_model_seg_image(get_s_pos(s_pos_json), p_pos.clone(),
                                                                model_parse_ag_full_image)
        model_parse_gen_image = pg.draw_parse_model_image(pg_output)
        st.image(model_parse_gen_image)
        st.session_state.pg_output = pg_output
        # model_parse_gen_image = pg.draw_parse_model_image(pg_output)

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

        c_pos, v_pos = get_c_pos_warp(st.session_state.key_points_tshirt.copy())

        im_backbone, im_left_up, im_right_up, im_left_low, im_right_low = generate_warp_images(
            np.asarray(Image.open(IMAGE_ADDRESS)),
            np.asarray(Image.open(CLOTH_ADDRESS)),
            v_pos.float(),
            torch.tensor(p_pos),
            ag_mask,
            skin_mask,
            st.session_state.pg_output

        )
        st.write(im_backbone.shape)
        Image.fromarray(im_backbone).save("im_backbone.jpg")
        # Load the images
        image1 = np.asarray(Image.open(IMAGE_ADDRESS))
        image2 = np.asarray(im_backbone, dtype = np.uint8)

        # Make sure both images have the same dimensions
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to create a mask for white pixels
        _, mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

        # Invert the mask
        mask = cv2.bitwise_not(mask)

        # Set white pixels to black in the original image using the mask
        image2[mask == 0] = [100, 100, 100]

        # Define the weight for each image
        alpha = 0.3  # Weight for the first image
        beta = 1.5  # Weight for the second image
        st.write("hiiii")
        # Blend the images
        blended_image = cv2.addWeighted(image1, alpha, image2, beta, 0)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image2)
        with col2:
            st.image(blended_image)
    if os.path.exists("im_backbone.jpg"):
        # Load the image using Pillow
        im_backbone = np.asarray(Image.open("im_backbone.jpg"))

        # Convert the image to grayscale using Pillow
        gray_image = np.asarray(Image.open("im_backbone.jpg").convert("L"))

        # Threshold the grayscale image to create a mask for white pixels
        _, mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

        # Invert the mask
        mask = cv2.bitwise_not(mask)
        # Make a copy of the original image to ensure it's writable
        im_backbone_copy = im_backbone.copy()

        # Set white pixels to black in the original image using the mask
        im_backbone_copy[mask == 0] = [0, 0, 0]
        st.write("Hi")
        value = streamlit_image_coordinates(
                Image.fromarray(im_backbone_copy),
                key="pil1")
        st.write(value)

        st.write(im_backbone[value["y"], value["x"]])








        #Image.fromarray(out_image).save('myra-app-main/predict/images/out_image.jpg')
        #Image.fromarray(out_mask).save('myra-app-main/predict/images/out_mask.jpg')
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
