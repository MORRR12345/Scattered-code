# train.py，主体训练代码
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from show import show
from model import AdvancedBatchLogger, build_model

# 加载数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 创建验证集
val_split = 0.2
split_idx = int(len(x_train) * (1 - val_split))
x_val = x_train[split_idx:]
y_val = y_train[split_idx:]
x_train = x_train[:split_idx]
y_train = y_train[:split_idx]

# 构建模型
model = build_model()
model.summary()

# 创建回调实例
advanced_logger = AdvancedBatchLogger()

# 训练模型
history = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val),
                    epochs=10,
                    batch_size=64,
                    callbacks=[advanced_logger])

# 显示训练结果
show(advanced_logger)

# 保存模型
model.save('mnist_cnn.h5')
print("模型已保存为 mnist_cnn.h5")

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n测试集准确率：{test_acc:.4f}")