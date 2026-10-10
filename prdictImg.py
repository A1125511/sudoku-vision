import cv2
import torch
from preprocess import preprocess_cell
from preprocess import bit_plane
from load_dataset import load_board

# [修正4] 圖片與對應 JSON 一起選，取代多份寫在程式裡、容易選錯的標籤。
image_path = 'img/img04.png'
label_path = 'img/img04.json'

# [修正4、5] 與訓練共用 load_board，同時取得正確標籤和同一套切格結果。
cell, label = load_board(image_path, label_path)
img = cv2.imread(image_path)
print(img.shape)
print(img.shape[0])
k = len(cell)


def test():
    cv2.imshow('img', img)
    for i in range(k):
        #cv2.imshow(f'number{i}_before', cell[i])
        processed = preprocess_cell(cell[i])
        cv2.imshow(f'number{i}_after', processed)
        print(processed.shape)

# test()

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(start_dim=1),
    torch.nn.Linear(8 * 16 * 16, 10)
)

weights = torch.load(
    "models/sudoku_cnn.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights)
model.eval()

predictions = []

with torch.no_grad():
    for i in range(len(cell)):
        processed = preprocess_cell(cell[i])

        image_tensor = torch.from_numpy(processed).float()
        image_tensor = image_tensor / 255.0

        image_tensor = image_tensor.unsqueeze(0)  # [1, 32, 32]
        x = image_tensor.unsqueeze(0)             # [1, 1, 32, 32]

        scores = model(x)
        predicted = scores.argmax(dim=1).item()

        predictions.append(predicted)

# 每行印出 9 格，方便和原圖對照
for row in range(9):
    print(predictions[row * 9:(row + 1) * 9])

wrong_answer = []

for row in range(9):
    for col in range(9):
        predicted = predictions[row * 9 + col]
        target = label[row][col]

        if predicted != target:
            print(
                "第", row + 1, "列，第", col + 1, "欄",
                "答案：", target,
                "預測：", predicted
            )


"""
gray = cv2.cvtColor(cell[0], cv2.COLOR_BGR2GRAY)
mask = bit_plane(gray)

count, component_map, stats, centroids = (
    cv2.connectedComponentsWithStats(mask, connectivity=8)
)

for component_id in range(1, count):
    x, y, width, height, area = stats[component_id]

    print(
        "區塊：", component_id,
        "位置：", x, y,
        "寬高：", width, height,
        "像素數：", area
    )
"""


cv2.waitKey()
cv2.destroyAllWindows()
