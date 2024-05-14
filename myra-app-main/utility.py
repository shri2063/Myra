from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import base64
import sys
import torch
sys.path.append('myra-app-main/predict')
from predict_pos_keypoints import adj_mx_from_edges,edges_c,edges_s,GCN_2
sys.path.append('myra-app-main/datasets')
import streamlit as st
import matplotlib.pyplot as plt


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

def get_c_pos_warp(c_pos_json) -> tuple:

    with open(c_pos_json, 'r') as f:
        c_pos = json.load(f)

        ck_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29,
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
def get_p_pos(key_points: torch.Tensor,s_pos_address) -> np.ndarray:
    adj_c = adj_mx_from_edges(32, edges_c, False)
    adj_s = adj_mx_from_edges(10, edges_s, False)
    kg_network = GCN_2(adj_c, adj_s, 160)
    kg_network.load_state_dict(
        torch.load('myra-app-main/checkpoints_pretrained/kg/step_299999.pt', map_location=torch.device('cpu')))
    kg_network.eval()


    key_points = key_points[1:,:]
    key_points = key_points/250
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
    #print("p pos", p_pos)
    p_pos = p_pos[0]
    with open("myra-app-main/predict/p_pos.json", "a") as file:
        json.dump(p_pos.tolist(), file)
    p_pos[:,0] = p_pos[:,0]*768
    p_pos[:, 1] = p_pos[:, 1] * 1024

    #print("p pos", p_pos.shape)
    return p_pos
