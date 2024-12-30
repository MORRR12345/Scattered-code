import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from config import map

def plot_voronoi():
    # 加载任务点数据
    task_pos = np.load('task_pos.npy')
    # 将任务点转换为适合Voronoi的格式 (n_points, 2)
    points = task_pos.T
    
    # 添加边界点以确保完整覆盖地图
    bound = 2  # 边界扩展系数
    x_bound = map.size_x/2 * bound
    y_bound = map.size_y/2 * bound
    
    # 在边界外围添加额外的点
    n_boundary_points = 8  # 每条边界添加的点数
    x = np.linspace(-x_bound, x_bound, n_boundary_points)
    y = np.linspace(-y_bound, y_bound, n_boundary_points)
    
    boundary_points = []
    # 添加上下边界点
    for i in x:
        boundary_points.append([i, y_bound])
        boundary_points.append([i, -y_bound])
    # 添加左右边界点
    for i in y:
        boundary_points.append([x_bound, i])
        boundary_points.append([-x_bound, i])
    
    # 合并原始点和边界点
    all_points = np.vstack((points, np.array(boundary_points)))
    
    # 创建Voronoi图
    vor = Voronoi(all_points)
    
    # 创建图形
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    
    # 绘制Voronoi区域
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), alpha=0.3)
    
    # 绘制Voronoi边界
    for simplex in vor.ridge_vertices:
        if simplex[0] >= 0 and simplex[1] >= 0:
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')
    
    # 只绘制原始任务点（不绘制边界点）
    plt.scatter(points[:, 0], points[:, 1], c='red', s=map.size_task)
    
    # 设置坐标轴范围
    plt.xlim(-map.size_x/2, map.size_x/2)
    plt.ylim(-map.size_y/2, map.size_y/2)
    
    plt.title('任务点Voronoi分区')
    plt.show()

if __name__ == "__main__":
    plot_voronoi() 