import tkinter as tk
from gomoku_env import GomokuEnv
from config import board, model
from dqn import ai_play

class GomokuGUI():

    # 初始化GomokuGUI类的实例。
    def __init__(self, master):
        self.master = master
        self.env = GomokuEnv()
        self.canvas = tk.Canvas(master, width=board.size_x*board.line_spacing, height=board.size_y*board.line_spacing)
        self.canvas.pack()
        self.player_id = 1
        self.canvas.bind("<Button-1>", self.click)
        self.draw_board()

    def draw_board(self):
        """
        在画布上绘制棋盘。
        """
        l_s = board.line_spacing
        for i in range(board.size_x):
            self.canvas.create_line(l_s/2, l_s/2 + i*l_s, 
                                    l_s/2 + (board.size_x-1)*l_s, l_s/2 + i*l_s)
        for i in range(board.size_y):
            self.canvas.create_line(l_s/2 + i*l_s, l_s/2, 
                                    l_s/2 + i*l_s, l_s/2 + (board.size_y-1)*l_s)

    def click(self, event):
        """
        处理鼠标点击事件，在棋盘上放置棋子，并更新棋盘状态。

        参数:
        event: 鼠标点击事件
        """
        l_s = board.line_spacing
        x, y = int(event.x // l_s), int(event.y // l_s)
        if 0 <= x < board.size_x and 0 <= y < board.size_y:
            done = self.play(x, y, self.player_id)
            self.player_id = 3 - self.player_id # 换玩家
            # 玩家输入结束，AI入场
            if model.player_AI and not done:
                self.master.after(500, self.play_ai)


    def play_ai(self):
        x1, y1 = ai_play(self.env.board, self.player_id, 0.0)
        self.play(x1, y1, self.player_id)
        self.player_id = 3 - self.player_id # 换玩家

    def play(self, x, y, player):
        _, win_id, done, _ = self.env.step((x, y), player)
        if win_id >= 0:
            self.draw_piece(x, y, player)
        if done:
            if win_id == 1: # 黑棋赢
                    message = "Black Win!"
            elif win_id == 2: # 白棋赢
                    message = "White Win!"
            else:
                message = "It's a Tie!"
            self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, text=message, font=("Arial", int(1.5*board.size_x+6)), fill="red")
            self.canvas.bind("<Button-1>", self.clear_board)
        return done
    
    # 在指定位置绘制棋子。
    def draw_piece(self, x, y, player):
        l_s = board.line_spacing
        color = "black" if player == 1 else "white"
        self.canvas.create_oval(l_s/2 + x*l_s - board.size_chess, l_s/2 + y*l_s - board.size_chess, 
                                l_s/2 + x*l_s + board.size_chess, l_s/2 + y*l_s + board.size_chess, fill=color)

    # 清空画布，开始新的一轮游戏。
    def clear_board(self, event):
        self.canvas.delete("all")
        self.draw_board()
        self.env.reset()
        self.player_id = 1
        self.canvas.unbind_all("<Button-1>")
        self.canvas.bind("<Button-1>", self.click)

def main():
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()