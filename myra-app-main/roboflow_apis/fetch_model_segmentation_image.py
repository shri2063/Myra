import numpy as np
from roboflow import Roboflow
from PIL import Image, ImageDraw
import numpy as np

rf = Roboflow(api_key="82dIk4wEDCIJrhZSCeq3")
project = rf.workspace().project("myra-1d8ni")
model = project.version(2).model


# infer on a local image
# print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
def predict_seg_img(model_image_address: str) -> Image:
    original_image = Image.open(model_image_address)
    model_seg = Image.new(mode='P', size=original_image.size, color=0)  ## Complete model Segmentation Image
    ag_mask = Image.new(mode='L', size=original_image.size, color=0)  ## tshirt mask
    skin_mask = Image.new(mode='L', size=original_image.size, color=0)  ## skin_mask
    draw = ImageDraw.Draw(model_seg)
    draw_ag_mask = ImageDraw.Draw(ag_mask)
    draw_skin_mask = ImageDraw.Draw(skin_mask)
    rb_json = model.predict(model_image_address, confidence=40).json()
    predictions = rb_json['predictions']
    print(predictions)
    for index, prediction in enumerate(predictions):

        points = prediction['points']
        points = [(int(point['x']), int(point['y'])) for point in points]
        if prediction['class'] == 'tshirt':
            draw_ag_mask.polygon(points, fill=255)
        if prediction['class'] == 'neck' or prediction['class'] == 'left_hand' or prediction['class'] == 'right_hand':
            draw_skin_mask.polygon(points, fill=255)

        draw.polygon(points, fill=index + 1)
    print("shape", model_seg.size)
    print("unique", np.unique(np.array(model_seg)))
    # Define a palette for the image
    palette = [
        255, 255, 255,  # white
        0, 255, 0,  # Green
        0, 0, 255,  # Blue
        255, 255, 0,  # Yellow
        255, 0, 255,  # Magenta
        0, 255, 255,  # Cyan
        128, 0, 128,  # Purple
        128, 128, 0  # Olive
    ]
    model_seg.putpalette(palette)
    model_seg.save("myra-app-main/predict/images/parse.png")
    ag_mask.save("myra-app-main/predict/images/ag_mask.png")
    skin_mask.save("myra-app-main/predict/images/skin_mask.png")
    return model_seg,ag_mask,skin_mask
    # model_seg.save("myra-app-main/predict/myra_v1/parse.png")


predict_seg_img("myra-app-main/upload_images/image.jpg")
