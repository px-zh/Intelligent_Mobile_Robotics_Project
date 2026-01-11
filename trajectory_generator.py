"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
#三次样条插值法（Cubic Spline）

class TrajectoryGenerator:
    def __init__(self, path):
        path = np.array(path, dtype=float)
        # 去除重复或几乎重合的点
        cleaned = [path[0]]
        for p in path[1:]:
            if np.linalg.norm(p - cleaned[-1]) > 1e-6:
                cleaned.append(p)
        self.path = np.array(cleaned)

        self.x, self.y, self.z = self.path[:,0], self.path[:,1], self.path[:,2]
        # 时间参数，用累计弧长作为时间
        diffs = np.diff(self.path, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        t = np.concatenate(([0.0], np.cumsum(seg_len)))
        # 避免浮点误差导致两个t相等，进一步保证严格递增
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + 1e-6
        self.t = t


        # 三次样条拟合
        self.cs_x = CubicSpline(self.t, self.x)
        self.cs_y = CubicSpline(self.t, self.y)
        self.cs_z = CubicSpline(self.t, self.z)

    def generate(self, num_points=500):
        t_new = np.linspace(self.t.min(), self.t.max(), num_points)
        x_fit = self.cs_x(t_new)
        y_fit = self.cs_y(t_new)
        z_fit = self.cs_z(t_new)
        return t_new, x_fit, y_fit, z_fit

    def plot(self, num_points=500):
        t_new, x_fit, y_fit, z_fit = self.generate(num_points)
        plt.figure(figsize=(12,8))
        plt.subplot(3,1,1)
        plt.plot(self.t, self.x, 'o', label='original x')
        plt.plot(t_new, x_fit, '-', label='Trajectory x')
        plt.xlabel('t (s)')
        plt.ylabel('x (m)')
        plt.legend()

        plt.subplot(3,1,2)
        plt.plot(self.t, self.y, 'o', label='original y')
        plt.plot(t_new, y_fit, '-', label='Trajectory y')
        plt.xlabel('t (s)')
        plt.ylabel('y (m)')
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(self.t, self.z, 'o', label='original z')
        plt.plot(t_new, z_fit, '-', label='Trajectory z')
        plt.xlabel('t (s)')
        plt.ylabel('z (m)')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    path = [[1, 2, 0], [2.4677935445576997, 1.7941422698448999, 1.4162290441486272],
            [1.8288821309099412, 2.5959495185190726, 3.1914465917985413],
            [1.4116457062945953, 3.7287058425423214, 3.373815065561532],
            [3.06518799840564, 4.854053571674452, 2.924492935306069],
            [4.670718356282364, 6.128342194618639, 2.8934965585622616],
            [4.895624821250256, 8.120530897623674, 3.3213998813289383],
            [6.553613061057961, 9.320428449962328, 3.203837725211193],
            [7.170441031691286, 11.261807271733467, 3.4342103039242025],
            [8.668379324325558, 12.617876944526579, 3.0881526295537225],
            [9.510047708717025, 14.36045973310037, 3.7645397640279405],
            [11.21515532768441, 14.111850368513986, 2.653998728505467],
            [12.814559455222586, 15.3526738285376, 2.3303267064385684],
            [14.631059067870662, 14.797273936895413, 3.1012742517444973],
            [15.637845319629905, 16.31686326507086, 2.163335172185562],
            [17.296908959371887, 16.191015756954197, 0.9657667006621529],
            [17.809555658604083, 17.5100054006453, 2.4489928066898043],
            [18, 18, 3]]

    traj = TrajectoryGenerator(path)
    traj.plot()


