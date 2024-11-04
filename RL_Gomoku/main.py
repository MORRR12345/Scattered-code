from dqn import train_dqn
from gui import main as gui_main
from config import board

# 检查输入数
def check_num(n):
    while True:
        num = int(input())
        if 0<= num <= n:
            break
        else:
            print("输入错误，请重新输入。")
    return num

if __name__ == "__main__":
    # print("模式：1、玩家对AI。2、玩家对玩家。3、训练AI。")
    # board.model = check_num(3)
    # # 模式1：玩家对AI
    # if board.model == 1:
    #     # print("请输入先手玩家：1、AI。2、玩家。")
    #     # first_player = check_12()
    #     gui_main()
    # # 模式2：玩家对玩家
    # elif board.model == 2:
    #     gui_main()
    # # 模式3：训练DQN模型
    # else:
    #     train_dqn()
    board.model = 1
    gui_main()