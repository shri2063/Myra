


import numpy as np
import cv2
import math
from torchvision import transforms
import torch
from PIL import  Image
import sys
sys.path.append('myra-app-main/models')
from networks_pg import ParseGenerator
import streamlit as st


PG_CHECK_POINT_DIR = "myra-app-main/checkpoints_pretrained/pg/step_9999.pt"
PARSE_SEG_IMAGE = "myra-app-main/predict/myra_v1/model_seg_image.png"
def draw_skeleton(sk_pos):
    sk_pos[:, 0] = sk_pos[:, 0] * 768
    sk_pos[:, 1] = sk_pos[:, 1] * 1024

    sk_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sk_Seq = [[0,1], [1,8], [1,9], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7]]

    stickwidth = 10

    jk_colors = [[255, 85, 0], [0, 255, 255], [255, 170, 0], [255, 255, 0], [255, 255, 0], [255, 170, 0], [85, 255, 0], [85, 255, 0], [0, 255, 255], [0, 255, 255]]
    sk_colors = [[255, 85, 0], [0, 255, 255], [0, 255, 255], [255, 170, 0], [255, 255, 0], [255, 255, 0], [255, 170, 0], [85, 255, 0], \
	          [85, 255, 0]]

    canvas = np.zeros((1024,768,3),dtype = np.uint8) # B,G,R order

    for i in range(len(sk_idx)):
        cv2.circle(canvas, (int(sk_pos[sk_idx[i]][0]),int(sk_pos[sk_idx[i]][1])), stickwidth, jk_colors[i], thickness=-1)

    for i in range(len(sk_Seq)):
        index = np.array(sk_Seq[i])
        cur_canvas = canvas.copy()
        Y = [sk_pos[index[0]][0],sk_pos[index[1]][0]]
        X = [sk_pos[index[0]][1],sk_pos[index[1]][1]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, sk_colors[i])
        canvas = cv2.addWeighted(canvas, 0, cur_canvas, 1, 0)

    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    canvas = transform(canvas)

    return canvas

def draw_cloth(ck_pos):
    ck_pos[:, 0] = ck_pos[:, 0] * 768
    ck_pos[:, 1] = ck_pos[:, 1] * 1024

    canvas = np.zeros((1024,768,3),dtype = np.uint8) # B,G,R order

    ck_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    ck_Seq = [[0,1], [1,2], [2,3], [3,4], \
        [4,31], [31,30], [30,29], [29,28], [28,27], [27,26], [26,25], [25,24], [24,23], [23,22], [22,21], [21,20], [20,19],\
        [19,18], [18,17], [17,16], [16,15], [15,14], [14,13], [13,12], [12,11], [11,10], [10,9], [9,8], [8,7], [7,6], [6,5], [5,0]]

    stickwidth = 10
    ck_colors = [255, 0, 0]

    for i in ck_idx:
        cv2.circle(canvas, (int(ck_pos[i][0]),int(ck_pos[i][1])), stickwidth, ck_colors, thickness=-1)

    for i in range(len(ck_Seq)):
        index = np.array(ck_Seq[i])
        cur_canvas = canvas.copy()
        Y = [ck_pos[index[0]][0], ck_pos[index[1]][0]]
        X = [ck_pos[index[0]][1], ck_pos[index[1]][1]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, ck_colors)
        canvas = cv2.addWeighted(canvas, 0, cur_canvas, 1, 0)

    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    canvas = transform(canvas)

    return canvas

def pred_to_onehot(prediction):
    size = prediction.shape
    prediction_max = torch.argmax(prediction, dim=1)
    oneHot_size = (size[0], 13, size[2], size[3])
    pred_onehot = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    pred_onehot = pred_onehot.scatter_(1, prediction_max.unsqueeze(1).data.long(), 1.0)
    return pred_onehot

def get_model_seg_image_hot_encoder(model_image):


    # Shorter side becomes 768 and larger side aligns based upon aspect ratio
    model_image = transforms.Resize(768)(model_image)
    model_image = np.asarray(model_image)

    # None adds dimesnion to first index
    if model_image.ndim > 2:
        im_parse_pil = model_image[:, :, 0]
    # unique_values = np.unique(im_parse_pil)
    # unique_values.sort()
    # print(("unique values", unique_values))
    # mapping = {label:i for i,label in enumerate(unique_values)}
    # im_parse_pil = np.vectorize(mapping.get)(im_parse_pil)
    parse = torch.from_numpy(np.array(model_image)[None]).long()
    parse_13 = torch.FloatTensor(13, 1024, 768).zero_()
    # Basically creates one hot encoding representation where eqach pixel value in the original image is represented as a one-hot vector along the zeroth dimension of parse_13
    parse_13 = parse_13.scatter_(0, parse, 1.0)
    parse_13 = parse_13[None]




    print(parse_13.shape)
    return parse_13
def gen_model_seg_image_hot_encoder(model_seg_Image: Image):


    # Shorter side becomes 768 and larger side aligns based upon aspect ratio
    im_parse_pil = transforms.Resize(768)(model_seg_Image)
    im_parse_pil = np.asarray(im_parse_pil)

    # None adds dimesnion to first index
    if im_parse_pil.ndim > 2:
        im_parse_pil = im_parse_pil[:, :, 0]
    unique_values = np.unique(im_parse_pil)
    unique_values.sort()
    print(("unique values", unique_values))
    mapping = {label: i for i, label in enumerate(unique_values)}
    im_parse_pil = np.vectorize(mapping.get)(im_parse_pil)
    # None adds dimension to first index
    parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

    parse_13 = torch.FloatTensor(13, 1024, 768).zero_()
    # Basically creates one hot encoding representation where eqach pixel value in the original image is represented as a one-hot vector along the zeroth dimension of parse_13
    parse_13 = parse_13.scatter_(0, parse, 1.0)
    parse_13 = parse_13[None]
    print(parse_13.shape)
    return parse_13


def parse_model_seg_image(s_pos: torch.Tensor, p_pos: torch.Tensor,model_parse_ag_full_image: Image):

    pg_network = ParseGenerator(input_nc=19, output_nc=13, ngf=64).to(torch.device('cpu'))

    pg_network.load_state_dict(torch.load(PG_CHECK_POINT_DIR, map_location=torch.device('cpu')))

    pg_network.eval()
    #print("s_pos", s_pos.shape)
    #print("p_pos", p_pos.shape)
    parse13_model_seg = get_model_seg_image_hot_encoder(model_parse_ag_full_image)
    sk_vis = draw_skeleton(s_pos)  # Model image with keypoints
    #print('sk_vis', sk_vis.shape)
    ck_vis = draw_cloth(p_pos)  # Tshirt image with keypoints
    #print('ck_vis', ck_vis.shape)

    norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    sk_input = torch.unsqueeze(norm_transform(sk_vis), dim=0)
    #print('sk_input', sk_input.shape)
    ck_input = torch.unsqueeze(norm_transform(ck_vis), dim=0)
    #print('ck_input', ck_input.shape)
    #print('parse13_model_seg', parse13_model_seg.shape)  # [1,13,768,1024] hot encoded vector of model segmentation image
    #print('parse ag non zero values', np.count_nonzero(parse13_model_seg))
    pg_input = torch.cat([parse13_model_seg, sk_input, ck_input], 1)

    pg_output = pg_network(pg_input)  # [1,13,768,1024] hot encoded vector of parse-generated model segmentation image
    #print('pg_output shape', pg_output.shape)
    pg_output = pred_to_onehot(pg_output).cpu()
    return pg_output, parse13_model_seg

def draw_parse_model_image(pg_output: torch.Tensor) -> Image:
    unique_colors = torch.randperm(256)[:39]
    colors = unique_colors.view(13, 3)
    # Convert hot_encoded_tensor to [13, 1024, 768] shape
    hot_encoded_tensor = pg_output.squeeze(0)
    # Apply the color palette to the one-hot encoded tensor
    parse_segmentation_image = torch.matmul(hot_encoded_tensor.permute(1, 2, 0), colors.float())

    # Convert the resulting tensor to uint8 and clamp values to [0, 255]
    parse_segmentation_image = parse_segmentation_image.byte().clamp(0, 255)
    parse_segmentation_image = parse_segmentation_image.to(torch.int32)
    parse_segmentation_image = parse_segmentation_image.detach().numpy()

    # segmentation_image = np.transpose(segmentation_image, axes = (1,2,0))
    parse_segmentation_image = parse_segmentation_image.astype(np.uint8)
    ## COLOR CODED SEGMENTATION IMAGE (using 'red' pixel value as color code)

    parse_segmentation_image = Image.fromarray(parse_segmentation_image)
    return parse_segmentation_image


