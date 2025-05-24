import tensorflow as tf
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 归一化像素值到 [0, 1]（可选）
train_images = train_images / 255.0
test_images = test_images / 255.0

# 添加通道维度（从 (28,28) 变为 (28,28,1)）
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]
# 获取第一张图片和标签
first_image = train_images[0]
first_label = train_labels[0]

# 显示图片
plt.figure()
plt.imshow(first_image.squeeze(), cmap='gray')  # 去掉通道维度并显示灰度图
plt.title(f"Label: {first_label}")
plt.colorbar()  # 显示颜色条（表示像素值）
plt.show()