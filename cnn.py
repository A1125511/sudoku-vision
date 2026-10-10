import torch
import random
from load_dataset import load_board, get_sample, load_dataset

num_epochs = 20
learning_rate = 0.001
# [修正2] 暫不宣告 batch_size、num_cnn：目前仍是每次一格、固定一層卷積。
# 等實作批次訓練或多層卷積後，再加入對應設定，避免改了變數卻沒有作用。
out_channels = 8
kernel_size = 3
padding = 1
Pooling = 2
shift = True

def create_model():
    conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
    )
    relu = torch.nn.ReLU()
    # [修正1] Pooling 使用自己的 2×2 視窗；不要共用卷積的 kernel_size=3。
    pool = torch.nn.MaxPool2d(kernel_size=Pooling, stride=Pooling)
    flatten = torch.nn.Flatten(start_dim=1)
    classifier = torch.nn.Linear(
        in_features=out_channels*16*16,
        out_features=10
    )
    model = torch.nn.Sequential(
        conv, relu, 
        pool, 
        flatten, 
        classifier
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer

def train_one_epoch(samples, model, loss_fn, optimizer):
    total_loss = 0.0

    for i in range(len(samples)):
        image_tensor, target = samples[i]

        y = torch.tensor([target], dtype=torch.long)
        x = image_tensor.unsqueeze(0)

        scores = model(x)
        optimizer.zero_grad()
        loss = loss_fn(scores, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(samples)

def evaluate(samples, index, model, show_errors=True):
    correct = 0
    image_tensor, target = samples[index]
    x = image_tensor.unsqueeze(0)
    scores = model(x)
    predicted = scores.argmax(dim=1).item()
    
    if predicted == target:
        correct += 1
    elif show_errors:
        # [新增準確率] 多張圖片合併後，先換回該圖片內的格子索引，列、欄才會保持在 1～9。
        cell_index = index % 81
        row = cell_index // 9 + 1
        col = cell_index % 9 + 1
        print("第", row, "列，第", col, "欄：","答案", target, "預測", predicted)

    return correct

def evaluate_dataset(samples, model, board_paths, show_errors=False):
    # [新增準確率] 共用同一套評估方式，計算整份資料答對幾格，以及每張圖片的準確率。
    # 呼叫前先 model.eval()，讓這一輪訓練完成的模型在評估期間保持固定。
    total_correct = 0
    board_accuracies = []

    with torch.no_grad():
        for board_index in range(len(board_paths)):
            if show_errors:
                print("圖片：", board_paths[board_index][0])

            correct = 0
            # load_dataset 按圖片順序合併資料；每張數獨固定有 81 格。
            start = board_index * 81
            end = start + 81
            for index in range(start, end):
                correct += evaluate(samples, index, model, show_errors)

            board_accuracies.append(correct / 81 * 100)
            total_correct += correct

    return total_correct, board_accuracies

def main():
    

    """
    cells, labels = load_board(
        "img/images.png",
        "img/images.json"
    )
    """
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)

    board_paths = [
        ("img/images.png", "img/images.json"),
        ("img/img03.png", "img/img03.json"),
        ("img/img04.png", "img/img04.json"),
    ]
    samples = load_dataset(board_paths, shift)    

    model, loss_fn, optimizer = create_model()
    validation_paths = [
        ("img/img2.png", "img/img2.json"),
    ]
    validation_samples = load_dataset(validation_paths, False)

    # [新增準確率] 記錄各輪結果，最後才能計算全部 epoch 的平均；訓練與驗證分開記錄。
    train_accuracy_history = []
    validation_accuracy_history = []
   
    for epoch in range(num_epochs):
        
        model.train()

        average_loss = train_one_epoch(
            samples=samples,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer
        )            

        print("第", epoch + 1, "輪，平均 loss =", average_loss)
        
        model.eval()
        # [新增準確率] 每輪結束後分別評估訓練、驗證資料，確保兩個準確率來自對應資料。
        train_correct, train_board_accuracies = evaluate_dataset(samples, model, board_paths)
        validation_correct, validation_board_accuracies = evaluate_dataset(
            validation_samples, model, validation_paths, show_errors=True
        )

        train_accuracy = train_correct / len(samples) * 100
        validation_accuracy = validation_correct / len(validation_samples) * 100
        train_accuracy_history.append(train_accuracy)
        validation_accuracy_history.append(validation_accuracy)

        print(f"訓練答對：{train_correct} / {len(samples)}，準確率：{train_accuracy:.2f} %")
        print(f"驗證答對：{validation_correct} / {len(validation_samples)}，準確率：{validation_accuracy:.2f} %")

    # [新增準確率] 以下各圖片結果來自最後一輪的模型；每張都有 81 格，因此圖片平均等於整體準確率。
    print("\n最後一輪各圖片的準確率：")
    for (image_path, _), accuracy in zip(board_paths, train_board_accuracies):
        print(f"訓練圖片 {image_path}：{accuracy:.2f} %")
    average_train_board_accuracy = sum(train_board_accuracies) / len(train_board_accuracies)
    print(f"訓練圖片平均準確率：{average_train_board_accuracy:.2f} %")

    for (image_path, _), accuracy in zip(validation_paths, validation_board_accuracies):
        print(f"驗證圖片 {image_path}：{accuracy:.2f} %")
    average_validation_board_accuracy = sum(validation_board_accuracies) / len(validation_board_accuracies)
    print(f"驗證圖片平均準確率：{average_validation_board_accuracy:.2f} %")

    # [新增準確率] 各輪百分比相加除以輪數，是訓練過程的摘要，與最後一輪準確率不同。
    average_train_accuracy = sum(train_accuracy_history) / len(train_accuracy_history)
    average_validation_accuracy = sum(validation_accuracy_history) / len(validation_accuracy_history)
    print(f"\n全部 {num_epochs} 輪平均訓練準確率：{average_train_accuracy:.2f} %")
    print(f"全部 {num_epochs} 輪平均驗證準確率：{average_validation_accuracy:.2f} %")

    model_name = "1conv_shift"  # 檔名標記；單層實驗填 "1conv"
    model_path = f"models/sudoku_{model_name}_seed{seed}.pth"
    torch.save(model.state_dict(), model_path)
    print("模型已保存：", model_path)
    #torch.save(model.state_dict(), "models/sudoku_cnn.pth")

    

if __name__ == "__main__":
    main()
