import numpy as np
from config import board

# 五子棋游戏环境类
class GomokuEnv:
    # 初始化游戏环境
    def __init__(self):
        self.board = np.zeros((board.size_x, board.size_y), dtype=int)  # 初始化棋盘
        # self.current_player = 1  # 初始化当前玩家为1

    # 重置游戏环境
    def reset(self):
        self.board = np.zeros((board.size_x, board.size_y), dtype=int)
        # self.current_player = 1
        return self.board

    # 执行一步游戏
    def step(self, action, player_id):
        x, y = action
        if self.board[x, y] != 0:  # 如果该位置已经有棋子，则动作无效
            return self.board, -1, False, {}  # Invalid move
        else:
            self.board[x, y] = player_id  # 在该位置下棋

            if self.check_winner(x, y):  # 如果下棋后当前玩家获胜
                return self.board, player_id, True, {}  # Current player wins
            
            if np.all(self.board != 0):  # 如果棋盘已满，则游戏平局
                return self.board, 0, True, {}  # Draw

            return self.board, 0, False, {}

    # 检查下棋后是否有玩家获胜
    def check_winner(self, x, y):
        # 检查所有方向上是否有连续五个棋子
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            if self.check_line(x, y, dx, dy):
                return True
        return False

    # 检查某一方向上是否有连续五个棋子
    def check_line(self, x, y, dx, dy):
        count = 0
        player = self.board[x, y]
        for d in range(-board.win_condition+1, board.win_condition):
            nx, ny = x + d * dx, y + d * dy
            if 0 <= nx < board.size_x and 0 <= ny < board.size_y and self.board[nx, ny] == player:
                count += 1
                if count == board.win_condition:
                    return True
            else:
                count = 0
        return False