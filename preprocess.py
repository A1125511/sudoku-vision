import cv2
import random
import numpy as np


def preprocess_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    top = bottom = int(height *0.2)
    left = right = int(width *0.2)
    cropped_img = gray[top:height-bottom, left:width-right]
    new_height, new_width = 32, 32
    resized_img = cv2.resize(cropped_img, (new_width, new_height))
    return resized_img

def bit_plane(cell):
    r,c = cell.shape
    x=np.zeros((r,c,8),dtype=np.uint8)
    for i in range(8):
        x[:, :, i] = 2 ** i
    r=np.zeros((r,c,8),dtype=np.uint8)
    for i in range(8):
        plane = cv2.bitwise_and(cell, x[:, :, i])
        plane[plane > 0] = 255
        r[:, :, i] = plane
    p = np.zeros(cell.shape, dtype=np.uint8)
    for i in range(7, 8):
        plane = (r[:, :, i] // 255) * (2 ** i)
        p |= plane
    mask = (1 - p // 128) * 255
    return mask

def shifted(cell):
    new_cell = np.ones_like(cell) * 255
    index_x = random.randint(-3,3)
    index_y = random.randint(-3,3)
    for i in range(len(new_cell)):
        for j in range(len(new_cell[i])):
            moved_x = index_x + j
            moved_y = index_y + i
            if (moved_x >= 0 and moved_x < len(cell[i])):
                if(moved_y >= 0 and moved_y < len(cell)):
                    new_cell[moved_y][moved_x] = cell[i][j]

    return new_cell
    


def main():
    pass

if __name__ == '__main__':
    main()