# 定义棋盘参数
class board():
    # 棋盘大小
    size_x = 3
    size_y = 3
    # 棋子大小
    size_chess = 15 
    # 线的宽局距
    line_spacing = 40
    # 胜利条件（连子多少颗算胜利）
    win_condition = 3

class model():
    # True为玩家对AI False为玩家对玩家
    player_AI = True

# 定义训练参数
class algorithm():
    linear = [128, 128]
    episodes = 100
    resume = False # 接着上次训练
    # 训练课程数
    train_course = 1