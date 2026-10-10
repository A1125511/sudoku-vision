import json
import cv2
import torch
from preprocess import preprocess_cell, shifted

def load_board(image_path, label_path):
    img = cv2.imread(image_path)

    with open(label_path, "r", encoding="utf-8") as file:
        labels = json.load(file)

    cells = []
    height, width = img.shape[:2]

    # [修正5] 各格的起點、終點分別計算，讓無法被 9 整除的像素也分配到格子裡。
    for i in range(9):
        for j in range(9):
            y0 = height * i // 9
            y1 = height * (i + 1) // 9
            x0 = width * j // 9
            x1 = width * (j + 1) // 9
            cells.append(img[y0:y1, x0:x1])

    return cells, labels

def get_sample(cells, label, index, shift):
    processed = preprocess_cell(cells[index])

    if shift:
        processed = shifted(processed)

    image_tensor = torch.from_numpy(processed)
    image_tensor = image_tensor.float()
    image_tensor = image_tensor/255.0
    image_tensor = image_tensor.unsqueeze(0)

    row = index // 9
    col = index % 9
    target = label[row][col]

    return image_tensor, target

def load_dataset(board_paths, shift):
    sample = []
    for i in range (len(board_paths)):
        image_path = board_paths[i][0]
        label_path = board_paths[i][1]

        cells, labels = load_board(image_path, label_path)

        for j in range(len(cells)):
            image_tensor, target = get_sample(cells, labels, j, shift)
            sample.append((image_tensor, target))

    return sample


def main():
    with open("img/images.json", "r", encoding="utf-8") as file:
        labels = json.load(file)

    print("列數：", len(labels))
    print("每列的格數：", [len(row) for row in labels])
    print("第一列：", labels[0])

    board_paths = [
        ("img/images.png", "img/images.json"),
        ("img/img03.png", "img/img03.json"),
        ("img/img04.png", "img/img04.json"),
    ]
    samples = load_dataset(board_paths, True)

    print("資料筆數：", len(samples))

    image_tensor, target = samples[0]
    print("第一筆圖片形狀：", image_tensor.shape)
    print("第一筆答案：", target)

    cells, labels = load_board(
        "img/images.png",
        "img/images.json"
    )

    print("小格數量：", len(cells))
    print("第一格形狀：", cells[0].shape)
    print("第一格答案：", labels[0][0])

    cv2.imshow("a", cells[0])
    hifted_img = shifted( cells[0])
    cv2.imshow("hifted", hifted_img)

    cells,_  = load_board("img/images.png","img/images.json")
    original = preprocess_cell(cells[0])
    cv2.imshow("original", original)

    shifted_img = shifted(original)
    cv2.imshow("shifted", shifted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
