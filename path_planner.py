"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""


#==============RRT3*D==============#
import numpy as np
import random
import time  # 统计时间


class Node3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.children = []  
        self.cost = 0.0  # 累计代价


class RRT3DStar:
    def __init__(self, env, start, goal,  # 三维坐标 (x,y,z)
                 max_iter=500,            # 最大迭代次数
                 goal_sample_rate=0.1,   # 采样目标点的概率
                 extend_length=1.0,       # 每次扩展的步长
                 goal_threshold=0.02,     # 判断到达目标的距离阈值
                 step_size_in=0.1       # 碰撞检测插值步长
                 ):
        self.env = env
        self.env_width = env.env_width
        self.env_length = env.env_length
        self.env_height = env.env_height

        self.start = Node3D(*start)
        self.goal = Node3D(*goal)
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.extend_length = extend_length
        self.goal_threshold = goal_threshold
        self.step_size = step_size_in
        self.nodes = [self.start]

        if self.is_invalid_point((start[0], start[1], start[2])):
            raise ValueError("起点位置无效（碰撞或超出边界）,请重新运行代码！"
                             "The starting position is invalid (collision or out of bounds). Please re-run the code!")
        if self.is_invalid_point((goal[0], goal[1], goal[2])):
            raise ValueError("终点位置无效（碰撞或超出边界）请重新运行代码！"
                             " The destination position is invalid (collision or out of bounds). Please re-run the code!")

    def is_invalid_point(self, point):
        # 边界判定
        return self.env.is_outside(point) or self.env.is_collide(point)

    def is_collision_line_free(self, p1, p2, step_size=0.1):
        # 检查线段是否无碰撞，通过在 p1 和 p2 之间插值多个点进行碰撞检测
        step_size = self.step_size
        vec = np.array([p2[0] - p1[0],
                        p2[1] - p1[1],
                        p2[2] - p1[2]])
        dist = np.linalg.norm(vec)
        num_checks = max(1, int(dist / step_size))

        for i in range(num_checks + 1):
            t = i / num_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            z = p1[2] + t * (p2[2] - p1[2])
            point = (x, y, z)
            if self.is_invalid_point(point):
                return False
        return True

    def get_random_node(self):
        # 以一定概率采样目标点
        if random.random() < self.goal_sample_rate:
            return Node3D(self.goal.x, self.goal.y, self.goal.z)
        return Node3D(
            random.uniform(0, self.env_width),
            random.uniform(0, self.env_length),
            random.uniform(0, self.env_height)
        )

    def steer(self, from_node, to_node):
        # 扩展一个固定步长
        vec = np.array([to_node.x - from_node.x,
                        to_node.y - from_node.y,
                        to_node.z - from_node.z])
        dist = np.linalg.norm(vec)
        if dist == 0:
            return None
        vec = vec / dist * min(self.extend_length, dist)
        new_node = Node3D(from_node.x + vec[0],
                          from_node.y + vec[1],
                          from_node.z + vec[2])
        return new_node

    def distance(self, n1, n2):
        return np.linalg.norm([n1.x - n2.x, n1.y - n2.y, n1.z - n2.z])

    def get_near_nodes(self, new_node):
        # 寻找邻域节点。

        n = len(self.nodes)
        if n <= 1:
            return []

        # 半径随节点数增加而减小，避免计算量过大
        radius = min(3.0, 1.5 * (np.log(n) / n) ** (1.0 / 3.0))
        # radius = 1.0


        near_nodes = []
        for node in self.nodes:
            if self.distance(node, new_node) <= radius:
                near_nodes.append(node)
        return near_nodes

    def choose_best_parent(self, new_node, near_nodes, default_parent):
        # 选择new_node的最优父节点：在所有邻域节点中，选取代价（cost + 距离）最小且无碰撞的一条连接。
        
        best_parent = default_parent
        best_cost = default_parent.cost + self.distance(default_parent, new_node)

        for node in near_nodes:
            if not self.is_collision_line_free((node.x, node.y, node.z),
                                               (new_node.x, new_node.y, new_node.z)):
                continue
            cost = node.cost + self.distance(node, new_node)
            if cost < best_cost:
                best_parent = node
                best_cost = cost

        new_node.parent = best_parent
        new_node.cost = best_cost
        best_parent.children.append(new_node)

    def update_subtree_cost(self, node):        
        # 当 rewiring 改变某个节点的父节点后，需要递归更新其子树中所有节点的 cost

        for child in node.children:
            child.cost = node.cost + self.distance(node, child)
            self.update_subtree_cost(child)

    def rewire(self, new_node, near_nodes):
        # 对邻域节点尝试重新连线，如果能降低其代价且路径无碰撞，则更新其父节点
        for node in near_nodes:
            if node == new_node.parent:
                continue

            new_cost = new_node.cost + self.distance(new_node, node)
            if new_cost + 1e-9 < node.cost:  # 加一点小量避免浮点数误差
                if self.is_collision_line_free((new_node.x, new_node.y, new_node.z),
                                               (node.x, node.y, node.z)):
                    if node.parent is not None:
                        node.parent.children.remove(node)

                    node.parent = new_node
                    new_node.children.append(node)

                    node.cost = new_cost
                    self.update_subtree_cost(node)

    def planning(self):
        start_time = time.time()
        iterations = 0

        best_goal_node = None  

        for _ in range(self.max_iter):
            iterations += 1

            rnd = self.get_random_node()
            nearest = min(self.nodes, key=lambda n: self.distance(n, rnd))
            new_node = self.steer(nearest, rnd)

            if new_node is None:
                continue

            if not self.is_collision_line_free((nearest.x, nearest.y, nearest.z),
                                               (new_node.x, new_node.y, new_node.z)):
                continue

            # 配置默认父节点
            default_parent = nearest
            #寻找邻域节点
            near_nodes = self.get_near_nodes(new_node)

            # 选择最优父节点
            if near_nodes:
                self.choose_best_parent(new_node, near_nodes, default_parent)
            else:
                new_node.parent = default_parent
                new_node.cost = default_parent.cost + self.distance(default_parent, new_node)

            self.nodes.append(new_node)
            if near_nodes:
                self.rewire(new_node, near_nodes)

            # if self.distance(new_node, self.goal) < self.goal_threshold:
            #     temp_goal = Node3D(self.goal.x, self.goal.y, self.goal.z)
            #     temp_goal.parent = new_node
            #     temp_goal.cost = new_node.cost + self.distance(new_node, temp_goal)

            #     if (best_goal_node is None) or (temp_goal.cost < best_goal_node.cost):
            #         best_goal_node = temp_goal

            if self.distance(new_node, self.goal) < self.goal_threshold:
                goal_node = Node3D(self.goal.x, self.goal.y, self.goal.z)
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + self.distance(new_node, goal_node)

                print("Path found!")
                print(f"Iterations: {iterations}")
                print(f"Planning time: {time.time() - start_time:.4f} seconds")
                return self.get_path(goal_node)


        end_time = time.time()

        if best_goal_node is None:
            print("No path found")
            print(f"Total iterations: {iterations}")
            print(f"Planning time: {end_time - start_time:.4f} seconds")
            return None

        print("Path found!")
        print(f"Total iterations: {iterations}")
        print(f"Planning time: {end_time - start_time:.4f} seconds")

        return self.get_path(best_goal_node)

    def get_path(self, goal_node=None):
        print("get_path_success!")
        path = []
        node = goal_node if goal_node is not None else self.goal
        while node is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        return path[::-1]
    


#==============AStar==============#
import numpy as np
import time

class AStar3D:
    def __init__(self, env, voxel_size=0.25):
        self.env = env
        self.res = voxel_size

        self.nx = int(env.env_width / voxel_size)
        self.ny = int(env.env_length / voxel_size)
        self.nz = int(env.env_height / voxel_size)

        # 26 邻域
        self.neighbors = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

    # 坐标变换
    def world_to_grid(self, p):
        return (
            int(p[0] / self.res),
            int(p[1] / self.res),
            int(p[2] / self.res),
        )

    def grid_to_world(self, idx):
        return (
            (idx[0] + 0.5) * self.res,
            (idx[1] + 0.5) * self.res,
            (idx[2] + 0.5) * self.res,
        )

    def in_bounds(self, idx):
        x, y, z = idx
        return 0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz

    def is_free(self, idx):
        # 栅格是否可通行：没出界和没碰撞
        if not self.in_bounds(idx):
            return False
        p = self.grid_to_world(idx)
        if self.env.is_outside(p):
            return False
        if self.env.is_collide(p):
            return False
        return True

    # 启发函数
    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def search(self, start, goal):
        t0 = time.time()

        start_idx = self.world_to_grid(start)
        goal_idx = self.world_to_grid(goal)

        if not self.is_free(start_idx):
            print("A*: 起点无效（越界或碰撞）")
            return None
        if not self.is_free(goal_idx):
            print("A*: 终点无效（越界或碰撞）")
            return None

        open_set = {start_idx: (0.0, None)}
        closed_set = set()
        parent_map = {}  # 记录 parent

        while open_set:
            # 找 f 最小的节点
            current_idx = min(
                open_set.keys(),
                key=lambda k: open_set[k][0] + self.heuristic(k, goal_idx)
            )
            current_g, current_parent = open_set[current_idx]

            # 记录 parent
            parent_map[current_idx] = current_parent

            # 到达目标
            if current_idx == goal_idx:
                path_idx = [current_idx]
                parent = current_parent
                while parent is not None:
                    path_idx.append(parent)
                    parent = parent_map.get(parent, None)

                path_idx.reverse()
                path = [self.grid_to_world(idx) for idx in path_idx]

                t1 = time.time()
                print(f"A*: Path found! {len(path)} points, time: {t1-t0:.4f} seconds")
                return path

            # 扩展邻居
            for dx, dy, dz in self.neighbors:
                nei = (current_idx[0] + dx,
                       current_idx[1] + dy,
                       current_idx[2] + dz)

                if nei in closed_set:
                    continue
                if not self.is_free(nei):
                    continue

                step_cost = np.linalg.norm(np.array(nei) - np.array(current_idx))
                g_new = current_g + step_cost

                if nei not in open_set or g_new < open_set[nei][0]:
                    open_set[nei] = (g_new, current_idx)

            # 当前节点移入 closed
            closed_set.add(current_idx)
            del open_set[current_idx]

        t1 = time.time()
        print(f"A*: No path found, time: {t1-t0:.4f} seconds")
        return None
