import time
import numpy as np

class Timer: #@save
    """记录多次运行时间"""

    def __init__(self):
        self.times = []  # 记录运行时间的列表
        self.start()  # 初始化时启动计时器

    def start(self):
        """启动计时器"""
        self.tik = time.time()  # 记录开始时间

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)  # 记录运行时间
        return self.times[-1]  # 返回最近一次运行时间

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)  # 计算平均运行时间

    def sum(self):
        """返回时间总和"""
        return sum(self.times)  # 计算总的运行时间

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()  # 计算累计运行时间并转换为列表返回
