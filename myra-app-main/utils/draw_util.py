import cv2
import numpy as np
import math

from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
def draw_cloth(ck_pos):
    ck_pos[:, 0] = ck_pos[:, 0] * 768
    ck_pos[:, 1] = ck_pos[:, 1] * 1024
    print("-------------------")
    canvas = np.zeros((1024,768,3),dtype = np.uint8) # B,G,R order

    ck_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    ck_Seq = [[0,1], [1,2], [2,3], [3,4], \
        [4,31], [31,30], [30,29], [29,28], [28,27], [27,26], [26,25], [25,24], [24,23], [23,22], [22,21], [21,20], [20,19],\
        [19,18], [18,17], [17,16], [16,15], [15,14], [14,13], [13,12], [12,11], [11,10], [10,9], [9,8], [8,7], [7,6], [6,5], [5,0]]

    stickwidth = 10
    ck_colors = [255, 0, 0]
    print("-------------------")
    for i in ck_idx:
        cv2.circle(canvas, (int(ck_pos[i][0]),int(ck_pos[i][1])), stickwidth, ck_colors, thickness=-1)

    print("Hey", ck_Seq)

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