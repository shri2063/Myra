from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import base64
import sys
import torch

sys.path.append('myra-app-main/predict')
from predict_pos_keypoints import adj_mx_from_edges, edges_c, edges_s, GCN_2
from typing import Tuple
sys.path.append('myra-app-main/datasets')
import streamlit as st
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
import string
import random
import cv2

#### CATALOG FUNCTIONS ###########
directory_models = r'myra-app-main/data/image'
directory_tshirts = r'myra-app-main/data/cloth'


def initialize(type):
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


####### KeyPoints marker functions##########
def update_point_over_image(edited_image: Image, node: str, value: dict, kp_arr: np.ndarray,
                            cover_area_pointer_list: list, cover_area_label_list: list, address: str):
    point_size = 5
    label_text = 'Point'
    draw = ImageDraw.Draw(edited_image)
    font_size = 16
    font = ImageFont.truetype("arial.ttf", font_size)
    text_width, text_height = 5, 5

    node = int(node)
    original_image = Image.open(address)
    # st.write(f"values earlier: Node: {node}--- {kp_arr[node][0]}--{kp_arr[node][1]}")
    if node < len(kp_arr):
        kp_arr[node][0] = value["x"]
        kp_arr[node][1] = value["y"]
        st.write(f"values after {kp_arr[node][0]}--{kp_arr[node][1]}")
        image_crop = original_image.crop(cover_area_label_list[node])
        edited_image.paste(image_crop, cover_area_label_list[node])
        image_crop = original_image.crop(cover_area_pointer_list[node])
        edited_image.paste(image_crop, cover_area_pointer_list[node])
    else:
        st.write(kp_arr.shape)
        value_to_append = np.array([value["x"], value["y"]], dtype=np.float32)
        kp_arr = np.append(kp_arr, [value_to_append], axis=0)

    draw.ellipse((kp_arr[node][0] - point_size,
                  kp_arr[node][1] - point_size,
                  kp_arr[node][0] + point_size,
                  kp_arr[node][1] + point_size),
                 fill='red')

    cover_area_pointer = (int(kp_arr[node][0]) - int(point_size),
                          int(kp_arr[node][1]) - int(point_size),
                          int(kp_arr[node][0]) + int(point_size) + 5,
                          int(kp_arr[node][1]) + int(point_size) + 5)
    cover_area_label = (int(kp_arr[node][0] + point_size + 5),
                        int(kp_arr[node][1] - text_height // 2),
                        int(kp_arr[node][0]) + int(point_size) + 5 + int(
                            text_width),
                        int(kp_arr[node][1]) - text_height // 2 + int(
                            text_height))
    if node < len(cover_area_pointer_list):
        cover_area_pointer_list[node] = cover_area_pointer
        cover_area_label_list[node] = cover_area_label
    else:

        cover_area_pointer_list = np.append(cover_area_pointer_list, cover_area_pointer)
        cover_area_label_list = np.append(cover_area_label_list, cover_area_label)

    text_x = kp_arr[node][0] + point_size + 5  # Adjust for spacing
    text_y = kp_arr[node][1] - text_height // 2
    st.write(text_y, text_x)
    draw.text((text_x, text_y), str(node), fill='red', font=font)
    st.image(edited_image)


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

def process_cutout(cutout_image:np.ndarray, model_image: np.ndarray):
    # Make sure both images have the same dimensions
    cutout_image_processed = cv2.resize(cutout_image, (model_image.shape[1], model_image.shape[0]))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(cutout_image_processed, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a mask for white pixels
    _, mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Set white pixels to black in the original image using the mask
    cutout_image_processed[mask == 0] = [100, 100, 100]
    return cutout_image_processed

def initialize_out_image_and_mask(out_image_arr: np.ndarray, model_image_arr : np.ndarray,
                                  mask_image_arr: np.ndarray, ag_mask_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    out_image_arr[ag_mask_arr == 255, :3] = 0
    out_image_arr[mask_image_arr == 11, :] = model_image_arr[mask_image_arr == 11, :]
    out_image_arr[mask_image_arr == 5, :] = model_image_arr[mask_image_arr == 5, :]
    out_image_arr[mask_image_arr == 6, :] = model_image_arr[mask_image_arr == 6, :]

    out_mask_arr = 255 - ag_mask_arr
    out_mask_arr = out_mask_arr.copy()

    out_mask_arr[mask_image_arr == 11] = 255
    out_mask_arr[mask_image_arr == 5] = 255
    out_mask_arr[mask_image_arr == 6] = 255
    return out_image_arr,out_mask_arr


def write_points_and_labels_over_image(arr: np.ndarray, image: Image, labels: {} = None) -> Image:
    point_size = 5
    label_text = 'Point'
    draw = ImageDraw.Draw(image)
    font_size = 16
    font = ImageFont.truetype("arial.ttf", 15)
    text_width, text_height = 5, 5
    point_color = 'red'

    st.write("hi6")
    if labels is not None:

        st.write("hi7")
        for id, point in enumerate(arr):
            draw.ellipse(
                (point[0] - point_size, point[1] - point_size, point[0] + point_size, point[1] + point_size),
                fill=point_color)
            text_x = point[0] + point_size + 5  # Adjust for spacing
            text_y = point[1] - text_height // 2

            draw.text((text_x, text_y), str(labels[id]), fill='red', font=font)
        return image
    st.write("hi8")
    for id, point in enumerate(arr):
        draw.ellipse(
            (point[0] - point_size, point[1] - point_size, point[0] + point_size, point[1] + point_size),
            fill=point_color)
        text_x = point[0] + point_size + 5  # Adjust for spacing
        text_y = point[1] - text_height // 2
        draw.text((text_x, text_y), str(id), fill='red', font=font)

    return image


def get_s_pos_string(s_pos_json) -> str:
    with open(s_pos_json, 'r') as f:
        s_pos = np.array(json.load(f)["people"][0]["pose_keypoints_2d"])

        s_pos = s_pos.tostring()
        s_pos = base64.b64encode(s_pos).decode('utf-8')
        return s_pos


def get_c_pos_string(c_pos_json) -> str:
    with open(c_pos_json, 'r') as f:
        c_pos = json.load(f)
        c_pos = np.array(c_pos["long"])
        c_pos = c_pos.tostring()
        c_pos = base64.b64encode(c_pos).decode('utf-8')
        return c_pos


def get_c_pos_warp(c_pos) -> tuple:
    ck_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 32]
    c_pos = c_pos[ck_idx, :] / 250

    c_pos[:, 0] = c_pos[:, 0] / 3
    c_pos[:, 1] = c_pos[:, 1] / 4
    v_pos = torch.tensor(c_pos)

    c_w = (c_pos[2][0] + c_pos[18][0]) / 2
    c_h = (c_pos[2][1] + c_pos[18][1]) / 2

    c_pos[:, 0] = c_pos[:, 0] - c_w
    c_pos[:, 1] = c_pos[:, 1] - c_h

    c_pos = torch.tensor(c_pos)
    return c_pos, v_pos


def get_s_pos(s_pos_address) -> torch.tensor:
    with open(s_pos_address, 'r') as f:
        s_pos = json.load(f)["people"][0]["pose_keypoints_2d"]
        sk_idx = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12]
        s_pos = np.resize(s_pos, (25, 3))[sk_idx, 0:2]
        s_pos[:, 0] = s_pos[:, 0] / 768
        s_pos[:, 1] = s_pos[:, 1] / 1024
        for l in range(10):
            if s_pos[l][0] == 0:
                if l in [0, 2, 5, 8, 9]:
                    s_pos[l, :] = s_pos[1, :]
                else:
                    s_pos[l, :] = s_pos[l - 1, :]

        s_pos = torch.from_numpy(s_pos)
        return s_pos


def get_p_pos(key_points: torch.Tensor, s_pos_address) -> np.ndarray:
    adj_c = adj_mx_from_edges(32, edges_c, False)
    adj_s = adj_mx_from_edges(10, edges_s, False)
    kg_network = GCN_2(adj_c, adj_s, 160)
    kg_network.load_state_dict(
        torch.load('myra-app-main/checkpoints_pretrained/kg/step_299999.pt', map_location=torch.device('cpu')))
    kg_network.eval()

    key_points = key_points[1:, :]
    key_points = key_points / 250
    key_points[:, 0] = key_points[:, 0] / 3
    key_points[:, 1] = key_points[:, 1] / 4

    c_w = (key_points[2][0] + key_points[18][0]) / 2
    c_h = (key_points[2][1] + key_points[18][1]) / 2

    key_points[:, 0] = key_points[:, 0] - c_w
    key_points[:, 1] = key_points[:, 1] - c_h
    key_points = torch.tensor(key_points)

    ## result['c_pos'] = result['c_pos].float()
    s_pos = get_s_pos(s_pos_address).float()
    key_points = key_points.float()

    p_pos = kg_network(key_points, s_pos).detach().numpy()
    # print("p pos", p_pos)
    p_pos = p_pos[0]
    with open("myra-app-main/predict/p_pos.json", "a") as file:
        json.dump(p_pos.tolist(), file)
    p_pos[:, 0] = p_pos[:, 0] * 768
    p_pos[:, 1] = p_pos[:, 1] * 1024

    # print("p pos", p_pos.shape)
    return p_pos


def generate_random_string(length):
    # Choose from all uppercase letters and digits
    characters = string.ascii_uppercase + string.digits
    # Generate a random string of given length
    return ''.join(random.choices(characters, k=length))

def convert_to_rgb(image, colormap):
    """
    Convert a labeled grayscale image to RGB using the provided colormap.
    """

    if len(image.shape) == 2 and len(np.unique(image)) > 2:

        rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for label, color in colormap.items():
            rgb_image[image == label] = color
        return rgb_image

    if len(image.shape) == 2 and len(np.unique(image)) == 2:
        image = np.stack((image,)*3, axis = -1)

    return image