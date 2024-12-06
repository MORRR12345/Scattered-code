import pyautogui
import time

# 设置在每项操作之间暂停的秒数
delay = 1

# 定义一个函数来点击特定位置的图标
def click_icon(icon_position):
    pyautogui.click(icon_position)
    # pyautogui.click(button='right') # 单击右键
    # pyautogui.doubleClick() # 双击左键
    time.sleep(delay)

# 获取鼠标位置
def get_mouse_position():
    position = pyautogui.position()
    print(f"Current mouse position: {position}")
    return position

# 定义一个函数来点击特定屏幕位置
def click_position(position):
    pyautogui.click(position)
    time.sleep(delay)

# 定义一个函数来使用键盘输入文字
def type_text(text):
    pyautogui.typewrite(text)
    time.sleep(delay)

# 示例用法

# 点击位于 (100, 200) 位置的图标
icon_position = (100, 200)
click_icon(icon_position)

# 点击位于 (300, 400) 位置的屏幕
position = (300, 400)
click_position(position)

# 键入文本 "你好，世界！"
text = "你好，世界！"
type_text(text)
