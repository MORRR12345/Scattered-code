import pyautogui
from pyautogui import KEYBOARD_KEYS

# 获取鼠标的当前位置
position = pyautogui.position()
print(f"Current mouse position: {position}")

# 单击左键
pyautogui.click()

# 单击右键
pyautogui.click(button='right')

# 双击左键
pyautogui.doubleClick()

# 点击特定位置
position = (100, 200)
pyautogui.click(position)

# 滚动鼠标向上
pyautogui.scroll(10)

# 滚动鼠标向下
pyautogui.scroll(-10)

# 按住键并输入文本
pyautogui.keyDown('shift')
pyautogui.typewrite('hello')
pyautogui.keyUp('shift')

# 按住键并输入文本（使用键盘常量）
pyautogui.keyDown(KEYBOARD_KEYS['ctrl'])
pyautogui.typewrite('a')
pyautogui.keyUp(KEYBOARD_KEYS['ctrl'])

# 按住键并输入文本（使用组合键）
pyautogui.hotkey('ctrl', 'a')

# 获取整个屏幕的截图
screenshot = pyautogui.screenshot()
screenshot.save('screenshot.png')

# 获取特定区域的截图
region = (100, 200, 300, 400)
screenshot = pyautogui.screenshot(region=region)
screenshot.save('region_screenshot.png')

# 移动鼠标到特定位置
position = (100, 200)
pyautogui.moveTo(position)

# 移动鼠标到特定位置并按住左键
pyautogui.mouseDown(button='left')
pyautogui.moveTo(300, 400)
pyautogui.mouseUp(button='left')

# 按住左键并拖动鼠标
pyautogui.mouseDown(button='left')
pyautogui.moveTo(300, 400)
pyautogui.mouseUp(button='left')