board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]

def printboard(board, given):
    for i in range(9):
        print("-------------------------------------")
        print(end="| ")
        for j in range(9):
            num = board[i][j]
            if num == 0:
                print("·", end=" | ")
            elif given[i][j]:
                print(f"\033[95m{num}\033[0m", end=" | ")  # 題目原有的數字
            else:
                print(f"\033[96m{num}\033[0m", end=" | ")  # 求解器填入的數字
        print()

def is_valid(board, row, col, num) -> bool:
    # 已經有數字的格子不能再填入
    if board[row][col] != 0:
        return False
    if num in board[row]:
        return False
    for i in range(9):
        if board[i][col] == num:
            return False
    startR, startC = 3*(row//3),3*(col//3)
    for i in range(3):
        for j in range(3):
            if board[startR+i][startC+j] == num:
                return False
    return True

def solve(board) -> bool:
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1,10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def validate_board(board) -> bool:
    if len(board) != 9:
        return False
    for i in range(9):
        if len(board[i]) != 9:
            return False
    for row in range(9):
        for col in range(9):
            num = board[row][col]
            if type(num) is not int or not 0 <= num <= 9:
                return False

            if num == 0:
                continue

            board[row][col] = 0
            valid = is_valid(board, row, col, num)
            board[row][col] = num

            if not valid:
                return False
    return True

if __name__ == "__main__":
    
    if validate_board(board):
        given = [[board[i][j] != 0 for j in range(9)] for i in range(9)]
        print("原始題目：")
        printboard(board, given)
        if solve(board):
            print("解題成功！")
            printboard(board, given)
        else:
            print("這題無解。")
    else:
        print("題目格式錯誤或數字有衝突")
