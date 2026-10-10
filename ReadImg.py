import cv2
import torch
from preprocess import preprocess_cell
from load_dataset import get_sample

img = cv2.imread('img/images.png')
print(img.shape)
print(img.shape[0])
cell = []
#k=0
label = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9],
]



"""
def test():
    cv2.imshow('img', img)
    for i in range(k):
        cv2.imshow(f'number{i}_before', cell[i])
        processed = preprocess_cell(cell[i])
        cv2.imshow(f'number{i}_after', processed)
        print(processed.shape)
"""


# test()

image_tensor, target = get_sample(0)

x = image_tensor.unsqueeze(0)

conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1,
)

output = conv(x)

relu = torch.nn.ReLU()
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

activated = relu(output)
pooled = pool(activated)

flatten = torch.nn.Flatten(start_dim=1)
classifier = torch.nn.Linear(
    in_features=8*16*16,
    out_features=10
)

features = flatten(pooled)
scores = classifier(features)

print("展平後：", features.shape)
print("分類分數形狀：", scores.shape)
print("分類分數：", scores)

print("輸入形狀：", x.shape)
print("輸出形狀：", output.shape)
print("ReLU 後：", activated.shape)
print("池化後：", pooled.shape)

print(image_tensor.shape)  # torch.Size([1, 32, 32])
print(image_tensor.dtype)  # torch.float32
print(image_tensor.min(), image_tensor.max())  # 都在 0～1 之間
print(target)             # 5

model = torch.nn.Sequential(
    conv, relu, pool, flatten, classifier
)

y = torch.tensor([target], dtype=torch.long)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
"""
for step in range(20):
    
    optimizer.zero_grad()       # 清除上一次計算的梯度

    scores = model(x)           # 用目前的權重重新預測
    loss = loss_fn(scores, y)   # 計算這次的損失

    loss.backward()             # 計算梯度
    optimizer.step()            # 根據梯度更新權重

    print("第", step + 1, "次，loss =", loss.item())
"""


for epoch in range(20):
    total_loss = 0.0

    for i in range(len(cell)):
        image_tensor, target = get_sample(i)
        x = image_tensor.unsqueeze(0)
        y = torch.tensor([target], dtype=torch.long)

        optimizer.zero_grad()

        scores = model(x)
        loss = loss_fn(scores, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(cell)
    print("第", epoch + 1, "輪，平均 loss =", average_loss)

model.eval()
correct = 0

with torch.no_grad():
    for i in range(len(cell)):
        image_tensor, target = get_sample(i)
        x = image_tensor.unsqueeze(0)

        scores = model(x)
        predicted = scores.argmax(dim=1).item()

        if predicted == target:
            correct += 1
        else:
            row = i // 9 + 1
            col = i % 9 + 1

            print(
                "猜錯的位置：第", row, "列，第", col, "欄",
                "正確答案：", target,
                "模型預測：", predicted
            )

print("答對：", correct, "/", len(cell))
print("訓練準確率：", correct / len(cell) * 100, "%")

torch.save(model.state_dict(), "models/sudoku_cnn.pth")

cv2.waitKey()
cv2.destroyAllWindows()


"""
image_tensor, target = get_sample(
    cells=cells, 
    label=labels,
    index=i
)

y = torch.tensor([target], dtype=torch.long)
x = image_tensor.unsqueeze(0)   #[1, 1, 32, 32]
output = conv(x)                #[1, 8, 32, 32]
activated = relu(output)        #[1, 8, 32, 32]
pooled = pool(activated)        #[1, 8, 16, 16]
features = flatten(pooled)      #[1, 2048]
scores = classifier(features)   #[1, 10]
scores = model(x)
optimizer.zero_grad()
loss = loss_fn(scores, y)
loss.backward()
optimizer.step()
""" 