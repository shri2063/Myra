# Set your Cloudinary credentials
# ==============================
from dotenv import load_dotenv

#load_dotenv()

# Import the Cloudinary libraries
# ==============================
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Import to format the JSON responses
# ==============================
import json

# Set configuration parameter: return "https" URLs by setting secure=True
# ==============================
config = cloudinary.config(cloud_name = "dit4rrqjd",
    api_key = "798112851553718",
    api_secret = "6NHEMBHenrU3xs3RlmETrI6mo9M", # Click 'View Credentials' below to copy your API secret
    secure=True)

# Log the configuration
# ==============================
print("****1. Set up and configure the SDK:****\nCredentials: ", config.cloud_name, config.api_key, "\n")


def uploadImage(location,name):
    # Upload the image and get its URL
    # ==============================

    # Upload the image.
    # Set the asset's public ID and allow overwriting the asset with new versions
    cloudinary.uploader.upload(location,
                               public_id=name, unique_filename=False, overwrite=True)

    # Build the URL for the image and save it in the variable 'srcURL'
    srcURL = cloudinary.CloudinaryImage(name).build_url()

    # Log the image URL to the console.
    # Copy this URL in a browser tab to generate the image on the fly.
    print("****2. Upload an image****\nDelivery URL: ", srcURL, "\n")
    return srcURL

def DeleteImage(name):
    # Upload the image and get its URL
    # ==============================

    # Upload the image.
    # Set the asset's public ID and allow overwriting the asset with new versions
    cloudinary.uploader.destroy(name)

    # Build the URL for the image and save it in the variable 'srcURL'
    #srcURL = cloudinary.CloudinaryImage(name).build_url()

    # Log the image URL to the console.
    # Copy this URL in a browser tab to generate the image on the fly.


def getAssetInfo():
    # Get and use details of the image
    # ==============================

    # Get image details and save it in the variable 'image_info'.
    image_info = cloudinary.api.resource("quickstart_butterfly")
    print("****3. Get and use details of the image****\nUpload response:\n", json.dumps(image_info, indent=2), "\n")

    # Assign tags to the uploaded image based on its width. Save the response to the update in the variable 'update_resp'.
    if image_info["width"] > 900:
        update_resp = cloudinary.api.update("quickstart_butterfly", tags="large")
    elif image_info["width"] > 500:
        update_resp = cloudinary.api.update("quickstart_butterfly", tags="medium")
    else:
        update_resp = cloudinary.api.update("quickstart_butterfly", tags="small")

    # Log the new tag to the console.
    print("New tag: ", update_resp["tags"], "\n")


def createImageTag():
    # Transform the image
    # ==============================

    # Create an image tag with transformations applied to the src URL.
    imageTag = cloudinary.CloudinaryImage("quickstart_butterfly").image(radius="max", effect="sepia")

    # Log the image tag to the console
    print("****4. Transform the image****\nTransfrmation URL: ", imageTag, "\n")


def main():
    #uploadImage()
    1 == 1


main();
