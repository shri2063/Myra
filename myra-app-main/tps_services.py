import torch
import numpy as np
import cv2
import streamlit as st
from torchvision.transforms import ToTensor, ToPILImage
import pyclipper
from PIL import Image



class TPS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h):
        """ grid """
        grid = torch.ones(1, h, w, 2)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        grid = grid.view(-1, h * w, 2)

        """ W, A """
        n, k = X.shape[:2]
        X = X
        Y = Y
        Z = torch.zeros(1, k + 3, 2)
        P = torch.ones(n, k, 3)
        L = torch.zeros(n, k + 3, k + 3)

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.linalg.solve(L,Z)[0]
        LU, pivots = torch.lu(L)
        Q = torch.lu_solve(Z.unsqueeze(1), LU, pivots)[0]
        W, A = Q[:, :k], Q[:, k:]

        """ U """
        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)

        """ P """
        n, k = grid.shape[:2]
        P = torch.ones(n, k, 3)
        P[:, :, 1:] = grid

        # grid = P @ A + U @ W
        grid = torch.matmul(P, A) + torch.matmul(U, W)
        return grid.view(-1, h, w, 2)


# this function appears to take a contour and a margin, performs an offset operation on the contour,
# and returns the resulting contour after offsetting.
def equidistant_zoom_contour(contour, margin):
    pco = pyclipper.PyclipperOffset()
    contour = contour[:, :]
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    if len(solution) == 0:
        solution = np.zeros((3, 2)).astype(int)
    else:
        solution = np.array(solution[0]).reshape(-1, 2).astype(int)

    return solution


def remove_background(s_mask, im):
    r_mask = s_mask.copy()
    for i in range(1024):
        for j in range(768):
            if im[i, j, 0] > 240 and im[i, j, 1] > 240 and im[i, j, 2] > 240:
                r_mask[i, j] = 0
    return r_mask


def dedup(source_pts, target_pts, source_center, target_center):
    old_source_pts = source_pts.tolist()
    old_target_pts = target_pts.tolist()
    idx_list = []
    new_source_pts = []
    new_target_pts = []
    for idx in range(len(old_source_pts)):
        if old_source_pts[idx] not in new_source_pts:
            if old_target_pts[idx] not in new_target_pts:
                new_source_pts.append(old_source_pts[idx])
                new_target_pts.append(old_target_pts[idx])
                idx_list.append(idx)

    if len(idx_list) == 2:
        new_source_pts = torch.cat([source_pts[idx_list], source_center], dim=0)[None, ...]
        new_target_pts = torch.cat([target_pts[idx_list], target_center], dim=0)[None, ...]

    elif len(idx_list) > 2:
        new_source_pts = source_pts[idx_list][None, ...]
        new_target_pts = target_pts[idx_list][None, ...]

    else:
        print("Less than 2 points are detected !")

    return new_source_pts, new_target_pts


def pred_to_onehot(prediction):
    size = prediction.shape

    prediction_max = torch.argmax(prediction, dim=1)
    oneHot_size = (size[0], 13, size[2], size[3])

    pred_onehot = torch.FloatTensor(torch.Size(oneHot_size)).zero_()

    pred_onehot = pred_onehot.scatter_(1, prediction_max.unsqueeze(1).data.long(), 1.0)
    return pred_onehot


def warp_part(group_id, ten_source, ten_target, ten_source_center, ten_target_center, ten_img):
    ten_source_p = ten_source[group_id]
    ten_target_p = ten_target[group_id]
    tps = TPS()
    ten_source_p, ten_target_p = dedup(ten_source_p, ten_target_p, ten_source_center, ten_target_center)
    warped_grid = tps(ten_target_p, ten_source_p, 768, 1024)
    ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0, False)
    out_img = np.array(ToPILImage()(ten_wrp[0].cpu()))
    return out_img


def paste_cloth(mask, image, tps_image, l_mask, r_mask, parse_13):
    out_image = image.copy()
    out_mask = mask.copy()
    # l_mask[(parse_13[3]).numpy() == 0] = 0
    # r_mask[(parse_13[3]).numpy() == 0] = 0

    out_mask[l_mask == 255] = 0
    out_mask[r_mask == 255] = 255

    out_image[l_mask == 255, :] = 0
    out_image[r_mask == 255, :] = tps_image[r_mask == 255, :]

    return out_mask, out_image


def generate_warp_images(image: np.ndarray, cloth: np.ndarray, source: torch.Tensor, target: torch.Tensor,
                         ag_mask: np.array, skin_mask: np.array, pg_output: np.ndarray):

    parse_13 = pred_to_onehot(pg_output)[0]

    ## Mask of tshirt in output image
    out_mask = ag_mask.copy()
    ## Mask of output image
    out_image = image.copy()
    ## Masking tshirt in output image , out_image = (h,w,3), ag_mask = (h,w)
    out_image[ag_mask == 0, :] = 0
    st.image(out_image)
    # Paste Skin
    new_skin_mask = skin_mask.copy()
    st.write("hiii")
    new_skin_mask[(parse_13[5] + parse_13[6] + parse_13[11]).numpy() == 0] = 0

    out_mask[new_skin_mask == 255] = 255
    out_image[new_skin_mask == 255, :] = image[new_skin_mask == 255, :]

    # Tshirt mask with skin
    # out_mask[new_skin_mask == 255] = 255
    # out_image[new_skin_mask == 255, :] = image[new_skin_mask == 255, :]

    # paste cloth
    group_backbone = [4, 3, 2, 1, 0, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31]
    group_left_up = [5, 6, 7, 12, 13, 14]
    group_left_low = [7, 8, 9, 10, 11, 12]
    group_right_up = [22, 23, 24, 29, 30, 31]
    group_right_low = [24, 25, 26, 27, 28, 29]

    ten_cloth = ToTensor()(cloth)
    # print("Cloth", cloth.shape)
    # print("Ten Cloth", ten_cloth.shape)
    ten_source = (source - 0.5) * 2
    ten_target = (target - 0.5) * 2
    ten_source_center = (0.5 * (ten_source[18] - ten_source[2]))[None, ...]  # [B x NumPoints x 2]
    ten_target_center = (0.5 * (ten_target[18] - ten_target[2]))[None, ...]  # [B x NumPoints x 2]

    # Whole Points TPS
    im_backbone = warp_part(
        group_backbone, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)

    im_left_up = warp_part(
        group_left_up, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)

    im_right_up = warp_part(
        group_right_up, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)

    im_left_low = warp_part(
        group_left_low, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)

    im_right_low = warp_part(
        group_right_low, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)

    st.image(im_backbone)
    st.image(im_left_up)
    st.image(im_right_up)
    st.image(im_left_low)
    st.image(im_right_low)


    return im_backbone, im_left_up, im_right_up, im_left_low, im_right_low


def generate_tps_st(image: np.ndarray, cloth: np.ndarray, source: torch.Tensor, target: torch.Tensor, ag_mask: np.array,
                    skin_mask: np.array, pg_output: np.ndarray):
    parse_13 = pred_to_onehot(pg_output)

    # print("parse 13", parse_13.shape)
    # print("image ", image.shape)
    # print("cloth ", cloth.shape)
    # print("source ", source.shape)
    # print("target ", target.shape)
    # print("ag_mask ", ag_mask.shape)
    # print("skin_mask ", skin_mask.shape)
    # print("parse_13 ", parse_13.shape)
    out_image, out_mask = generate_warp_images(image, cloth, source, target, ag_mask, skin_mask, parse_13[0])
