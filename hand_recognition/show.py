# show.py，用于绘制实现结果
import numpy as np
import matplotlib.pyplot as plt 
import pickle

# 处理中文字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   

def save_logger_data(advanced_logger, filename='logger_data.pkl'):
    """保存logger数据到文件"""
    data = {
        'batch_losses': advanced_logger.batch_losses,
        'batch_accuracies': advanced_logger.batch_accuracies
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"数据已保存到 {filename}")

def load_logger_data(filename='logger_data.pkl'):
    """从文件加载logger数据"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def show(advanced_logger):
    """可视化训练指标"""
    # save_logger_data(advanced_logger, 'logger_data_Adagrad.pkl') # 保存数据
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(advanced_logger.batch_losses, label='训练损失')
    plt.title('训练损失变化')
    plt.xlabel('Batch')
    plt.ylabel('Loss')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(advanced_logger.batch_accuracies, label='训练准确率')
    plt.title('训练准确率变化')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

def main():
    """对比实验"""
    logger_data_Adam = load_logger_data('logger_data_Adam.pkl')
    logger_data_Adagrad = load_logger_data('logger_data_Adagrad.pkl')
    logger_data_RMSprop = load_logger_data('logger_data_RMSprop.pkl')
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(logger_data_Adam['batch_losses'], label='Adam优化器')
    plt.plot(logger_data_Adagrad['batch_losses'], label='Adagrad优化器')
    plt.plot(logger_data_RMSprop['batch_losses'], label='RMSprop优化器')
    plt.title('训练损失变化')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(logger_data_Adam['batch_accuracies'], label='Adam优化器')
    plt.plot(logger_data_Adagrad['batch_accuracies'], label='Adagrad优化器')
    plt.plot(logger_data_RMSprop['batch_accuracies'], label='RMSprop优化器')
    plt.title('训练准确率变化')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()