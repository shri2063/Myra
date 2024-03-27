import cv2
import torch
from dataset_tps import get_result
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor,ToPILImage
import numpy as np
import pyclipper





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
#this function appears to take a contour and a margin, performs an offset operation on the contour,
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
            if im[i, j, 0] >240 and im[i, j, 1] >240 and im[i, j, 2] >240:
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
def draw_part(group_id, ten_source, ten_target, ten_source_center, ten_target_center, ten_img):
    ten_source_p = ten_source[group_id]
    ten_target_p = ten_target[group_id]
    poly = ten_target[group_id].numpy()
    ## Not sure why we are alingning like below
    poly[:, 0] = (poly[:, 0] * 0.5 + 0.5) * 768
    poly[:, 1] = (poly[:, 1] * 0.5 + 0.5) * 1024
    image = ten_img.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    #plt.imshow(image)
    #x, y = zip(*poly)
    #plt.scatter(x, y, color="blue", s=10)
    #plt.show()
    l_mask = np.zeros((1024, 768))
    s_mask = np.zeros((1024,768))
    new_poly = equidistant_zoom_contour(poly, -5)
    # Masks  of left, right should and thsirt backbones
    cv2.fillPoly(l_mask, np.int32([poly]), 255)
    cv2.fillPoly(s_mask, np.int32([new_poly]), 255)
    tps = TPS()
    ten_source_p, ten_target_p = dedup(ten_source_p, ten_target_p, ten_source_center, ten_target_center)
    print('ten_source_p', ten_source_p.shape)
    warped_grid = tps(ten_target_p, ten_source_p, 768, 1024)
    print("warped Grid Shape", warped_grid.shape)
    ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0, False)
    out_img = np.array(ToPILImage()(ten_wrp[0].cpu()))
    r_mask = remove_background(s_mask, out_img)
    fig,axes = plt.subplots(2,4)
    axes[0][0].imshow(l_mask, cmap = 'gray')
    axes[0][0].set_title('L mask')
    axes[0][1].imshow(s_mask, cmap = 'gray')
    axes[0][1].set_title('S mask')
    axes[0][2].imshow(r_mask, cmap='gray')
    axes[0][2].set_title('R mask')
    axes[0][3].imshow(out_img, cmap='gray')
    axes[0][3].set_title('Out Image')
    plt.show()
    # image = warped_grid[0].cpu().numpy()
    # image = np.transpose(image, (1,2,0))


    return out_img, l_mask, s_mask, r_mask

def paste_cloth(mask, image, tps_image, l_mask, r_mask, parse_13):
    out_image = image.copy()
    out_mask = mask.copy()
    l_mask[(parse_13[3]).numpy() == 0] = 0
    r_mask[(parse_13[3]).numpy() == 0] = 0

    out_mask[l_mask==255] = 0
    out_mask[r_mask==255] = 255

    out_image[l_mask==255, :] = 0
    out_image[r_mask==255, :] = tps_image[r_mask==255, :]

    return out_mask, out_image

def generate_repaint(image, cloth, source, target, ag_mask, skin_mask, parse_13):
    out_mask = ag_mask.copy()
    out_image = image.copy()
    out_image[ag_mask == 0, :] = 0
    image_ag = out_image.copy()

    # paste skin
    new_skin_mask = skin_mask.copy()
    new_skin_mask[(parse_13[5] + parse_13[6] + parse_13[11]).numpy() == 0] = 0

    out_mask[new_skin_mask == 255] = 255
    out_image[new_skin_mask == 255, :] = image[new_skin_mask == 255, :]

    # paste cloth
    group_backbone = [4, 3, 2, 1, 0, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31]
    group_left_up = [5, 6, 7, 12, 13, 14]
    group_left_low = [7, 8, 9, 10, 11, 12]
    group_right_up = [22, 23, 24, 29, 30, 31]
    group_right_low = [24, 25, 26, 27, 28, 29]

    ten_cloth = ToTensor()(cloth)

    ten_source = (source - 0.5) * 2
    ten_target = (target - 0.5) * 2
    ten_source_center = (0.5 * (ten_source[18] - ten_source[2]))[None, ...]  # [B x NumPoints x 2]
    ten_target_center = (0.5 * (ten_target[18] - ten_target[2]))[None, ...]  # [B x NumPoints x 2]

    # Whole Points TPS
    im_backbone, l_mask_backbone, s_mask_backbone, r_mask_backbone = draw_part(
         group_backbone, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_left_up, l_mask_left_up, s_mask_left_up, r_mask_left_up = draw_part(
         group_left_up, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_right_up, l_mask_right_up, s_mask_right_up, r_mask_right_up = draw_part(
         group_right_up, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_left_low, l_mask_left_low, s_mask_left_low, r_mask_left_low = draw_part(
         group_left_low, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_right_low, l_mask_right_low, s_mask_right_low, r_mask_right_low = draw_part(
         group_right_low, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)

    if r_mask_backbone.sum() / s_mask_backbone.sum() < 0.9:
        r_mask_backbone = s_mask_backbone

    out_mask, out_image = paste_cloth(out_mask, out_image, im_backbone, l_mask_backbone, r_mask_backbone, parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_left_up, l_mask_left_up, r_mask_left_up, parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_left_low, l_mask_left_low, r_mask_left_low, parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_right_up, l_mask_right_up, r_mask_right_up, parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_right_low, l_mask_right_low, r_mask_right_low, parse_13)

    return out_image, out_mask, image_ag
def generate_repaint_1(image, cloth, source, target, ag_mask, skin_mask, parse_13):
    ## Mask of tshirt in output image
    out_mask = ag_mask.copy()
    ## Mask of output image
    out_image = image.copy()
    ## Masking tshirt in output image , out_image = (h,w,3), ag_mask = (h,w)
    out_image[ag_mask == 0, :] = 0
    # Paste Skin
    new_skin_mask = skin_mask.copy()
    new_skin_mask[(parse_13[5] + parse_13[6] + parse_13[11]).numpy() == 0] = 0
    #plt.title("New Skin Mask")
    #plt.imshow(new_skin_mask, cmap = 'gray')
    #plt.show()

    # Tshirt mask with skin
    out_mask[new_skin_mask == 255] = 255
    out_image[new_skin_mask == 255, :] = image[new_skin_mask == 255, :]

    # paste cloth
    group_backbone = [4, 3, 2, 1, 0, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31]
    group_left_up = [5, 6, 7, 12, 13, 14]
    group_left_low = [7, 8, 9, 10, 11, 12]
    group_right_up = [22, 23, 24, 29, 30, 31]
    group_right_low = [24, 25, 26, 27, 28, 29]

    ten_cloth = ToTensor()(cloth)
    print("Cloth", cloth.shape)
    print("Ten Cloth", ten_cloth.shape)
    ten_source = (source - 0.5) * 2
    ten_target = (target - 0.5) * 2
    ten_source_center = (0.5 * (ten_source[18] - ten_source[2]))[None, ...]  # [B x NumPoints x 2]
    ten_target_center = (0.5 * (ten_target[18] - ten_target[2]))[None, ...]  # [B x NumPoints x 2]

    # Whole Points TPS
    im_backbone, l_mask_backbone, s_mask_backbone, r_mask_backbone = draw_part(
        group_backbone, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_left_up, l_mask_left_up, s_mask_left_up, r_mask_left_up = draw_part(
         group_left_up, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_right_up, l_mask_right_up, s_mask_right_up, r_mask_right_up = draw_part(
         group_right_up, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_left_low, l_mask_left_low, s_mask_left_low, r_mask_left_low = draw_part(
         group_left_low, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_right_low, l_mask_right_low, s_mask_right_low, r_mask_right_low = draw_part(
         group_right_low, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)

    if r_mask_backbone.sum() / s_mask_backbone.sum() < 0.9:
        r_mask_backbone = s_mask_backbone

    out_mask, out_image = paste_cloth(out_mask, out_image, im_backbone, l_mask_backbone, r_mask_backbone, parse_13)
    plt.title("Backbone Added")
    plt.imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    plt.show()
    out_mask, out_image = paste_cloth(out_mask, out_image, im_left_up, l_mask_left_up, r_mask_left_up, parse_13)
    plt.title("Left Upper")
    plt.imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    plt.show()
    out_mask, out_image = paste_cloth(out_mask, out_image, im_left_low, l_mask_left_low, r_mask_left_low, parse_13)
    plt.title("Left Lower")
    plt.imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    plt.show()
    out_mask, out_image = paste_cloth(out_mask, out_image, im_right_up, l_mask_right_up, r_mask_right_up, parse_13)
    plt.title("right upper")
    plt.imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    plt.show()
    out_mask, out_image = paste_cloth(out_mask, out_image, im_right_low, l_mask_right_low, r_mask_right_low, parse_13)
    plt.title("right lower")
    plt.imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    plt.show()



def generation_tps():
    result = get_result();
    #print(result)
    image = result['image']
    cloth = result['cloth']
    ## mask of the tshirt in output image
    ag_mask = result['ag_mask']
    skin_mask = result['skin_mask']
    parse_13 = result['parse_13']
    ## (32,2) key pointers of Source Tshirt
    source = result['v_pos'].float()
    ## (32,2) key pointers of Target Tshirt
    target = result['e_pos'].float()
    out_image, out_mask = generate_repaint_1(image, cloth, source, target, ag_mask, skin_mask, parse_13)


generation_tps()
