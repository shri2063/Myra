from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import json
import cv2


# skeleton pos
def get_s_pos() -> torch.tensor:
    with open('myra-app-main/data/00006_00/00006_00_keypoints.json', 'r') as f:
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



# cloth pos
def get_c_pos():
    with open('myra-app-main/data/00006_00/cloth_landmark_json.json', 'r') as f:
        c_pos = json.load(f)


        ck_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
              30, 31, 32]
        c_pos = np.array(c_pos["long"])[ck_idx, :]




        c_pos[:, 0] = c_pos[:, 0] / 3
        c_pos[:, 1] = c_pos[:, 1] / 4
        v_pos = torch.tensor(c_pos)

        c_w = (c_pos[2][0] + c_pos[18][0]) / 2
        c_h = (c_pos[2][1] + c_pos[18][1]) / 2

        c_pos[:, 0] = c_pos[:, 0] - c_w
        c_pos[:, 1] = c_pos[:, 1] - c_h

        c_pos = torch.tensor(c_pos)
        return c_pos,v_pos
# parse
def get_parse_img():
    parse = Image.open('myra-app-main/data/00006_00/parse.png')
    parse = transforms.Resize(768)(parse)
    parse = torch.from_numpy(np.array(parse)[None]).long()
    return parse;


def get_image():
    image = cv2.imread('myra-app-main/data/00006_00/image.jpg', cv2.IMREAD_COLOR)

    x = np.array(transforms.Resize(768)(Image.fromarray(image)))

    return np.array(transforms.Resize(768)(Image.fromarray(image)))


def get_cloth():
    cloth = cv2.imread('myra-app-main/data/00006_00/cloth.jpg', cv2.IMREAD_COLOR)
    return np.array(transforms.Resize(768)(Image.fromarray(cloth)))


def get_ag_mask():
    ag_mask = 255 - cv2.imread('myra-app-main/data/00006_00/ag_mask.png', cv2.IMREAD_GRAYSCALE)
    return np.array(transforms.Resize(768)(Image.fromarray(ag_mask)))


def get_skin_mask():
    skin_mask = cv2.imread('myra-app-main/data/00006_00/skin_mask.png', cv2.IMREAD_GRAYSCALE)
    return np.array(transforms.Resize(768)(Image.fromarray(skin_mask)))


def get_parse_model_hot_encode():
    im_parse_pil_big = Image.open('seg.png')  # 768x1024
    print(np.array(im_parse_pil_big).shape)

    # Shorter side becomes 768 and larger side aligns based upon aspect ratio
    im_parse_pil = transforms.Resize(768)(im_parse_pil_big)
    im_parse_pil = np.asarray(im_parse_pil)

    # None adds dimesnion to first index
    if im_parse_pil.ndim > 2:
        im_parse_pil = im_parse_pil[:,:,0]
    unique_values = np.unique(im_parse_pil)
    unique_values.sort()
    print(("unique values", unique_values))
    mapping = {label:i for i,label in enumerate(unique_values)}
    im_parse_pil = np.vectorize(mapping.get)(im_parse_pil)
    parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
    parse_13 = torch.FloatTensor(13, 1024, 768).zero_()
    # Basically creates one hot encoding representation where eqach pixel value in the original image is represented as a one-hot vector along the zeroth dimension of parse_13
    parse_13 = parse_13.scatter_(0, parse, 1.0)
    parse_13 = parse_13[None]
    print(parse_13.shape)
    return parse_13




# Cloth Position
'''
def get_v_pos():
    with open('myra-app-main/data/00006_00/cloth_landmark_json.json','r') as f:
        c_pos = json.load(f)
    ck_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    # create a subset of c_pos as per index mentioned
    c_pos = np.array(c_pos["long"])[ck_idx, :]
    c_pos[:, 0] = c_pos[:, 0] / 3
    c_pos[:, 1] = c_pos[:, 1] / 4
    v_pos = torch.tensor(c_pos)
    return v_pos
'''


# estimated cloth position
def get_e_pos():
    with open('myra-app-main/data/00006_00/paired-ck-point.json', 'r') as f:
        e_pos = json.load(f)
        e_pos = np.array(e_pos["keypoints"])
        e_pos = torch.tensor(e_pos)
        return e_pos


def get_dataset_dict():
    c_pos, v_pos = get_c_pos()
    result = {
        'im_name': '00006_00',  # image name
        'cloth_name': '00006_00',  # cloth name
        'mix_name': '00006_0006',
        'image': get_image(),  # target image
        'cloth': get_cloth(),  # cloth image raw numpy array bgr
        'v_pos': v_pos,  # cloth keypoints position raw
        'e_pos': get_e_pos(),  # estimated cloth keypoints position
        'ag_mask': get_ag_mask(),
        'skin_mask': get_skin_mask(),
        'parse_ag': get_parse_model_hot_encode(),
        'c_pos': c_pos,
        's_pos': get_s_pos(),
        'parse': get_parse_img()
    }
    return result



