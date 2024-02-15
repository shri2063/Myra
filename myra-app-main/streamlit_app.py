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
                 "photoshoots by leveraging AI-generated models. Here's the concept: We begin with an image of a mannequin showcasing the fashion product under ideal "
            "lighting conditions. This image, along with your prompt detailing the desired attributes of the final model, is fed into our Myra AI system. "
            "From there, our model swiftly crafts the perfect image tailored to your specifications in no time. </span>",
            unsafe_allow_html=True)
        st.write(
            "<span style='font-family: Roboto,sans-serif'>We have listed below few examples of results obtained from Myra AI .The mannequin images below were retrieved from the internet from different sites. "
            "pinterest, shutterstock, istockphoto. "
            " Myra AI could "
            "fit AI models within this dress, maintaining the dress outline, tone, and appearance.</span>",
            unsafe_allow_html=True)

        st.write("")
        M1 = image_select(
            label="WHITE CREAM COLOUR FULL TSHIRT ON BLUE JEANS (https://www.pinterest.com/pin/30891947430775019)",
            images=[
                "myra-app-main/images/M1/input.png", "myra-app-main/images/M1/1-3.png",
                "myra-app-main/images/M1/2-2.png", "myra-app-main/images/M1/3-5.png", "myra-app-main/images/M1/4-3.png", "myra-app-main/images/M1/5-3.png"

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
        st.write("<span style='font-family: Roboto,sans-serif'>PLEASE CLICK ON ANY ABOVE IMAGE FOR A CLOSER LOOK.</span>",
                 unsafe_allow_html=True)
        if M1 != "images/M1/input.png":
            # Load the images
            image1 = Image.open("myra-app-main/images/M1/input.png")
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
            label="RED COLLAR DOTTED TSHIRT ON BLUE JEANS (https://www.shutterstock.com/image-photo/fulllength-male-mannequin-dressed-tshirt-jeans-1067987750)",
            images=[
                "myra-app-main/images/M6/input.png", "myra-app-main/images/M6/1-4.png",
                "myra-app-main/images/M6/2.png", "myra-app-main/images/M6/3-5.png", "myra-app-main/images/M6/4-3.png", "myra-app-main/images/M6/5-2.png"

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
        if M6 != "images/M6/input.png":
            # Load the images
            image1 = Image.open("myra-app-main/images/M6/input.png")
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
            label="White half sleeves tshirt on blue jeans (https://www.istockphoto.com/photo/full-length-male-mannequin-gm1289535860-385180516)",
            images=[
                "myra-app-main/images/M7/input.png", "myra-app-main/images/M7/1-2.png",
                "myra-app-main/images/M7/2-2.png", "myra-app-main/images/M7/3-2.png", "myra-app-main/images/M7/4-3.png",

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
        if M7 != "images/M7/input.png":
            # Load the images
            image1 = Image.open("myra-app-main/images/M7/input.png")
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
