import json
import torch
import numpy as np
def get_c_pos():
    with open('myra-app-main/data/00006_00/cloth_landmark_json.json', 'r') as f:
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

def get_s_pos() -> torch.tensor:
    with open('myra-app-main/upload_images/00006_00_keypoints.json', 'r') as f:
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