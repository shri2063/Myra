import replicate
import streamlit as st
import requests
import zipfile
import io
from utils import icon
from streamlit_image_select import image_select
from PIL import Image

# UI configurations
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

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()


def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application, 
    including the form for user inputs and the resources section.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Yo fam! Start here ↓**", icon="👋🏾")
            with st.expander(":rainbow[**Refine your output here**]"):
                # Advanced Settings (for the curious minds!)
                width = st.number_input("Width of output image", value=1024)
                height = st.number_input("Height of output image", value=1024)
                num_outputs = st.slider(
                    "Number of images to output", value=1, min_value=1, max_value=4)
                scheduler = st.selectbox('Scheduler', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                       'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
                num_inference_steps = st.slider(
                    "Number of denoising steps", value=50, min_value=1, max_value=500)
                guidance_scale = st.slider(
                    "Scale for classifier-free guidance", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
                prompt_strength = st.slider(
                    "Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of infomation in image)", value=0.8, max_value=1.0, step=0.1)
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

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)

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

        return submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt


def main_page(submitted: bool, width: int, height: int, num_outputs: int,
              scheduler: str, num_inference_steps: int, guidance_scale: float,
              prompt_strength: float, refine: str, high_noise_frac: float,
              prompt: str, negative_prompt: str) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
        width (int): Width of the output image.
        height (int): Height of the output image.
        num_outputs (int): Number of images to output.
        scheduler (str): Scheduler type for the model.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Scale for classifier-free guidance.
        prompt_strength (float): Prompt strength when using img2img/inpaint.
        refine (str): Refine style to use.
        high_noise_frac (float): Fraction of noise to use for `expert_ensemble_refiner`.
        prompt (str): Text prompt for the image generation.
        negative_prompt (str): Text prompt for elements to avoid in the image.
    """
    if submitted:
        with st.status('👩🏾‍🍳 Whipping up your words into art...', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            try:
                # Only call the API if the "Submit" button was pressed
                if submitted:
                    # Calling the replicate API to get the image
                    with generated_images_placeholder.container():
                        all_images = []  # List to store all generated images
                        output = replicate.run(
                            REPLICATE_MODEL_ENDPOINTSTABILITY,
                            input={
                                "prompt": prompt,
                                "width": width,
                                "height": height,
                                "num_outputs": num_outputs,
                                "scheduler": scheduler,
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                                "prompt_stregth": prompt_strength,
                                "refine": refine,
                                "high_noise_frac": high_noise_frac
                            }
                        )
                        if output:
                            st.toast(
                                'Your image has been generated!', icon='😍')
                            # Save generated image to session state
                            st.session_state.generated_image = output

                            # Displaying the image
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(image, caption="Generated Image 🎈",
                                             use_column_width=True)
                                    # Add image to the list
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Save all generated images to session state
                        st.session_state.all_images = all_images

                        # Create a BytesIO object
                        zip_io = io.BytesIO()

                        # Download option for each image
                        with zipfile.ZipFile(zip_io, 'w') as zipf:
                            for i, image in enumerate(st.session_state.all_images):
                                response = requests.get(image)
                                if response.status_code == 200:
                                    image_data = response.content
                                    # Write each image to the zip file with a name
                                    zipf.writestr(
                                        f"output_file_{i+1}.png", image_data)
                                else:
                                    st.error(
                                        f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}", icon="🚨")
                        # Create a download button for the zip file
                        st.download_button(
                            ":red[**Download All images**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
                status.update(label="✅ images generated!",
                              state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Encountered an error: {e}', icon="🚨")

    # If not submitted, chill here 🍹
    else:
        pass




    # Gallery display for inspo
    if 1 == 1:

        st.write(
            "<span style='font-family: Roboto, sans-serif;'>At Myra, our vision is to inject a sense of magic and creativity into e-commerce fashion "
                 "photoshoots by leveraging AI-generated models and props.</span>",
            unsafe_allow_html=True)
        st.write(
            "<span style='font-family: Roboto,sans-serif'>The mannequin images below were retrieved from the internet."
            "For more information on these images, please visit the resource section. Our Finetuned model could "
            "fit a model within this dress, maintaining the dress outline, tone, and appearance.</span>",
            unsafe_allow_html=True)

        st.write("<span style='font-family: Roboto,sans-serif'>PLEASE CLICK ON ANY IMAGE FOR A CLOSER LOOK.</span>",
            unsafe_allow_html=True)
        M1 = image_select(
            label="White cream colour full sleeves tshirt on blue jeans",
            images=[
                "images/M1/input.png", "images/M1/1.png",
                "images/M1/2.png", "images/M1/3.png", "images/M1/4.png", "images/M1/5.png"

            ],
            captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                      "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                      "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
                      "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      ],
            use_container_width=False
        )

        if M1 != "images/M1/input.png":
            # Load the images
            image1 = Image.open("images/M1/input.png")
            image2 = Image.open(M1)

            # Create two columns for displaying images side by side
            col1, col2 = st.columns(2)

            # Display the images in the columns
            with col1:
                st.image(image1, caption='Input Image', use_column_width=True)

            with col2:
                st.image(image2, caption='Output Image', use_column_width=True)


        st.write("")
        st.write("")
        st.write("")
        M6 = image_select(
            label="Red collar tshirt on blue jeans",
            images=[
                "images/M6/input.png", "images/M6/1.png",
                "images/M6/2.png", "images/M6/3.png", "images/M6/4.png", "images/M6/5.png"

            ],
            captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                      "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                      "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
                      "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      ],
            use_container_width=False
        )
        if M6 != "images/M6/input.png":
            # Load the images
            image1 = Image.open("images/M6/input.png")
            image2 = Image.open(M6)

            # Create two columns for displaying images side by side
            col1, col2 = st.columns(2)

            # Display the images in the columns
            with col1:
                st.image(image1, caption='Input Image', use_column_width=True)

            with col2:
                st.image(image2, caption='Output Image', use_column_width=True)




        st.write("")
        st.write("")
        st.write("")
        M7 = image_select(
            label="White half sleeves tshirt on blue jeans ",
            images=[
                "images/M7/input.png", "images/M7/1.png",
                "images/M7/2.png", "images/M7/3.png", "images/M7/4.png",

            ],
            captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                      "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                      "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
                      "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      ],
            use_container_width=False
        )
        if M7 != "images/M7/input.png":
            # Load the images
            image1 = Image.open("images/M7/input.png")
            image2 = Image.open(M7)

            # Create two columns for displaying images side by side
            col1, col2 = st.columns(2)

            # Display the images in the columns
            with col1:
                st.image(image1, caption='Input Image', use_column_width=True)

            with col2:
                st.image(image2, caption='Output Image', use_column_width=True)

        st.write("")
        st.write("")
        st.write("")

        M5 = image_select(
            label="blue jeans ",
            images=[
                "images/M5/input.png", "images/M5/1.jpg",
                "images/M5/2.png", "images/M5/3.jpg", "images/M5/5.png",

            ],
            captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                      "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                      "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
                      "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      ],
            use_container_width=False
        )
        if M5 != "images/M5/input.png":
            # Load the images
            image1 = Image.open("images/M5/input.png")
            image2 = Image.open(M5)

            # Create two columns for displaying images side by side
            col1, col2 = st.columns(2)

            # Display the images in the columns
            with col1:
                st.image(image1, caption='Input Image', use_column_width=True)

            with col2:
                st.image(image2, caption='Output Image', use_column_width=True)

        st.write("")
        st.write("")
        st.write("")

        gen_img = image_select(
            label="General High Quality Images obtained from Parent AI Model Before we finetuned it ",
            images=[
                "myra-app-main/gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
                "gallery/friends.png","gallery/wizard.png","gallery/puppy.png","gallery/cheetah.png", "gallery/viking.png"
            ],
            captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                      "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                      "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
                      "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      "A cheetah mother nurses her cubs in the tall grass of the Serengeti. The early morning sun beams down through the grass. National Geographic photography by Frans Lanting",
                      "A close-up portrait of a bearded viking warrior in a horned helmet. He stares intensely into the distance while holding a battle axe. Dramatic mood lighting, digital oil painting",
                      ],
            use_container_width=False
        )
        if gen_img:
            st.image(Image.open(gen_img))


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt = configure_sidebar()
    try:
        main_page(submitted, width, height, num_outputs, scheduler, num_inference_steps,
              guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt)
    except Exception as e:
        1 == 1


if __name__ == "__main__":
    main()
