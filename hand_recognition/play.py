# play.py，实现可交互式的验证
import tkinter as tk
from tkinter import Canvas, Label, Button
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 加载保存的模型
model = tf.keras.models.load_model('mnist_cnn.h5')
# 处理中文字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")
        
        # 创建画布
        self.canvas = Canvas(root, width=480, height=480, bg='black')
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        # 创建按钮
        self.clear_btn = Button(root, text="清除", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=1, padx=10)
        
        self.predict_btn = Button(root, text="识别", command=self.predict_digit)
        self.predict_btn.grid(row=1, column=1, padx=10)
        
        # 创建概率显示区域
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_plot.get_tk_widget().grid(row=1, column=0, padx=10)
        
        # 初始化绘图变量
        self.image = Image.new('L', (480, 480), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        
    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='white', outline='white')
        self.draw.ellipse([x-10, y-10, x+10, y+10], fill='white')
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (480, 480), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.ax.clear()
        self.canvas_plot.draw()
        
    def predict_digit(self):
        # 预处理图像
        img = self.image.resize((28, 28))  # 缩放为28x28
        img_array = np.array(img) / 255.0  # 归一化
        img_array = np.expand_dims(img_array, axis=-1)  # 添加通道维度
        img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
        
        # 预测
        predictions = model.predict(img_array)
        predicted_digit = np.argmax(predictions)
        
        # 显示概率分布
        self.ax.clear()
        bars = self.ax.bar(range(10), predictions[0], color='skyblue')
        bars[predicted_digit].set_color('red')
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel('数字')
        self.ax.set_ylabel('概率')
        self.ax.set_title(f'预测结果: {predicted_digit}')
        self.ax.set_ylim([0, 1])
        self.canvas_plot.draw()

# 创建主窗口
root = tk.Tk()
app = DigitRecognizer(root)
root.mainloop()