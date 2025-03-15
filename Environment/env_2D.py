import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

class Agent:
    """智能体动力学模型"""
    def __init__(self, pos, agent_type):
        self.pos = np.array(pos, dtype=np.float32)
        self.type = agent_type
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)
        self.orientation = 0.0
        self.trajectory = [self.pos.copy()]
        
        # 动力学参数
        self.max_speed = 2.5
        self.max_acc = 1.8
        self.damping = 0.95

class GridEnvironment:
    """二维物理环境"""
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.agents = {}  # {agent_id: Agent}
        self.next_id = 0
        self.dt = 0.1
        
    def add_agent(self, pos, agent_type):
        """添加智能体返回数字ID"""
        agent_id = self.next_id
        self.agents[agent_id] = Agent(pos, agent_type)
        self.next_id += 1
        return agent_id
    
    def apply_acc(self, agent_id, acc):
        """应用加速度到指定智能体"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        accel = np.clip(acc, -agent.max_acc, agent.max_acc)
        agent.acc = np.array(accel)
    
    def update_world(self):
        """更新物理状态"""
        for agent_id, agent in self.agents.items():
            # 更新速度
            agent.vel = agent.vel * agent.damping + agent.acc * self.dt
            speed = np.linalg.norm(agent.vel)
            
            # 速度限制
            if speed > agent.max_speed:
                agent.vel = agent.vel / speed * agent.max_speed
                
            # 更新位置
            new_pos = agent.pos + agent.vel * self.dt
            
            # 边界处理
            if new_pos[0] < 0 or new_pos[0] > self.width:
                agent.vel[0] *= -0.7
            if new_pos[1] < 0 or new_pos[1] > self.height:
                agent.vel[1] *= -0.7
                
            agent.pos = np.clip(new_pos, [0, 0], [self.width, self.height])
            
            # 更新方向
            if np.linalg.norm(agent.vel) > 0.1:
                agent.orientation = np.degrees(np.arctan2(agent.vel[1], agent.vel[0]))
            
            agent.trajectory.append(agent.pos.copy())

class GridVisualizer:
    """可视化控制系统"""
    def __init__(self, env, type_colors=None):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.type_colors = type_colors or {'default': 'blue'}
        
        # 动画控制
        self.animation_running = False
        self.ani = None
        self._init_controls()
        self._setup_axes()
        
        # 回调函数列表
        self.update_callbacks = []
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
    
    def _init_controls(self):
        """初始化控制面板"""
        plt.subplots_adjust(bottom=0.15)
        ax_play = plt.axes([0.4, 0.02, 0.1, 0.05])
        ax_reset = plt.axes([0.51, 0.02, 0.1, 0.05])
        
        self.btn_play = Button(ax_play, '▶')
        self.btn_reset = Button(ax_reset, '↻')
        
        self.btn_play.on_clicked(self._toggle_animation)
        self.btn_reset.on_clicked(self._reset)
    
    def _setup_axes(self):
        """设置画布参数"""
        self.ax.clear()
        self.ax.set_xlim(-1, self.env.width+1)
        self.ax.set_ylim(-1, self.env.height+1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Multi-Agent Control Demo")
    
    def _draw_agent(self, agent, color):
        """绘制单个智能体"""
        # 绘制轨迹
        if len(agent.trajectory) > 1:
            traj = np.array(agent.trajectory)
            self.ax.plot(traj[:,0], traj[:,1], color=color, alpha=0.3, lw=1)
        
        # 绘制本体
        body = plt.Circle(agent.pos, 0.4, fc=color, ec='black', zorder=3)
        self.ax.add_patch(body)
        
        # 绘制速度箭头
        arrow = plt.Arrow(agent.pos[0], agent.pos[1],
                         agent.vel[0]*0.5, agent.vel[1]*0.5,
                         width=0.3, color='red')
        self.ax.add_patch(arrow)
        
        # 显示状态信息
        info = f"V: {np.linalg.norm(agent.vel):.1f}\nA: {np.linalg.norm(agent.acc):.1f}"
        self.ax.text(agent.pos[0]+0.5, agent.pos[1]+0.5, info, fontsize=8)
    
    def _animation_update(self, frame):
        """动画更新函数"""
        if self.animation_running:
            # 执行所有回调函数
            for callback in self.update_callbacks:
                callback()
            
            self.env.update_world()
            self._update_display()
        return []
    
    def _update_display(self):
        """更新画面"""
        self.ax.clear()
        self._setup_axes()
        for agent_id, agent in self.env.agents.items():
            color = self.type_colors.get(agent.type, '#666666')
            self._draw_agent(agent, color)
    
    def _toggle_animation(self, event):
        """切换运行动画"""
        self.animation_running = not self.animation_running
        self.btn_play.label.set_text('⏸' if self.animation_running else '▶')
    
    def _reset(self, event):
        """重置环境"""
        for agent in self.env.agents.values():
            agent.pos = agent.trajectory[0].copy()
            agent.vel = np.zeros(2)
            agent.acc = np.zeros(2)
            agent.trajectory = [agent.pos.copy()]
        self._update_display()
    
    def _on_key(self, event):
        """键盘控制回调"""
        if not self.animation_running:
            return
        
        # 默认控制第一个智能体
        if self.env.agents:
            agent_id = 0  # 第一个智能体ID
            
            # 方向键控制
            control_map = {
                'up': [0, 1],
                'down': [0, -1],
                'left': [-1, 0],
                'right': [1, 0]
            }
            
            if event.key in control_map:
                self.env.apply_acc(agent_id, control_map[event.key])
    
    def register_callback(self, callback):
        """注册更新回调"""
        self.update_callbacks.append(callback)
    
    def start(self):
        """启动动画系统"""
        self.ani = animation.FuncAnimation(
            self.fig,
            self._animation_update,
            interval=50,
            blit=False
        )
        plt.show()

# 示例使用 ######################################################
if __name__ == "__main__":
    # 初始化环境
    env = GridEnvironment(width=12, height=8)
    
    # 添加三个不同类型的智能体
    robot_id = env.add_agent([2, 2], 'robot')
    drone_id = env.add_agent([10, 6], 'drone')
    car_id = env.add_agent([6, 4], 'car')
    
    # 创建可视化器
    visualizer = GridVisualizer(env, {
        'robot': '#FF6B6B',
        'drone': '#4ECDC4',
        'car': '#45B7D1'
    })
    
    # 定义自动控制函数（示例：无人机圆周运动）
    def auto_control_drone():
        t = plt.matplotlib.dates.date2num(plt.matplotlib.dates.datetime.datetime.now())
        accel = [1.5 * np.sin(2*np.pi*t), 1.5 * np.cos(2*np.pi*t)]
        env.apply_acc(drone_id, accel)
    
    # 定义自动控制函数（示例：汽车随机运动）
    def auto_control_car():
        random_accel = np.random.uniform(-1, 1, 2)
        env.apply_acc(car_id, random_accel)
    
    # 注册自动控制回调
    visualizer.register_callback(auto_control_drone)
    visualizer.register_callback(auto_control_car)
    
    # 启动系统
    visualizer.start()