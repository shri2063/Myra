import numpy as np
from roboflow import Roboflow
from PIL import  Image, ImageDraw
import numpy as np
rf = Roboflow(api_key="82dIk4wEDCIJrhZSCeq3")
project = rf.workspace().project("myra-1d8ni")
model = project.version(2).model

# infer on a local image
#print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
def predict():
    original_image = Image.open("myra-app-main/upload_images/image.jpg")
    mask = Image.new(mode='P', size=original_image.size, color=0)
    draw = ImageDraw.Draw(mask)
    rb_json = model.predict("myra-app-main/upload_images/image.jpg", confidence=40).json()
    predictions = rb_json['predictions']
    for index,prediction in enumerate(predictions):

        points = prediction['points']
        points = [ (int(point['x']), int(point['y'])) for point in points]

        draw.polygon(points, fill=index + 1)
    print("shape", mask.size)
    print("unique", np.unique(np.array(mask)))
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
    mask.putpalette(palette)
    mask.save("myra-app-main/predict/images/model_seg_image.png")


predict()