"""
Cooperative Coverage Environment II (Four Directions, 48 robots, evaluation)
Problem Formulation: N mobile robots should cover N landmarks, we will get a positive reward if a robot covers one
                     landmark, and we will get a negative reward if different robots cover the same landmark.

Author: Yifan Hu, SEU
Date: 09/13/2022
Environment Details:  1) system dynamics: p_{t+1} = p_t + v_t*dt
                      2) no obstacle is considered in this version
                      3) action: [v_i^x,v_i^y]
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors


class CooperativeCoverage:
    def __init__(self, args):
        # individual robot parameters
        self.dt = 0.1  # sampling time interval
        self.dim = 2  # 2-D robot
        self.space = 1.  # space for individual robot
        self.r = 0.1  # robot radius
        self.safe_dis = 0.3  # safe distance between desired positions

        # problem parameters
        self.n_agents = args.n_agents
        self.degree = args.degree  # number of communication neighbors
        self.tar_num = args.tar_num  # number of observed closest targets
        self.rob_num = args.rob_num  # number of observed closest robots
        self.feature_num = (2*self.tar_num + self.rob_num)*self.dim  # info of closest targets and neighboring robots

        self.global_pos = np.zeros((self.n_agents, self.dim))
        self.adj_mat = np.zeros((self.n_agents, self.n_agents))  # adjacent matrix

        # initial landmark settings (fixed)
        self.ref_tar_pos = np.array([0., 0.])  # center of all landmarks
        self.tar_row_num = 7
        self.tar_column_num = 7
        self.n_targets = self.tar_row_num*self.tar_column_num - 1
        self.tar_pos = np.zeros((self.n_targets, self.dim))  # target positions
        self.tar_r = 0.05  # target radius

        # task indices
        self.counter = 0  # current step in an episode
        self.cover_rate = 0  # if 4 of 5 goal positions are covered, then self.cover_rate = 0.8
        self.collision = 0  # collision times in an episode
        self.cover_reward = 0  # coverage reward
        self.collision_reward = 0  # collision reward
        self.closest_rob_dist = None
        self.closest_rob_idx = None

        # figure parameters
        self.fig = None
        self.start_ax = None
        self.tar_ax = None  # draw target position
        self.rob_ax = None  # draw robot position
        self.comm_ax = None  # draw communication link
        self.start_pos = None  # record the start positions
        self.tar_circle = self.draw_circle(self.tar_r)
        self.rob_circle = self.draw_circle(self.r)
        self.order = 0

        print("Number of landmarks: {}".format(self.n_targets))
        print("Number of agents: {}".format(self.n_agents))
        print("Dim of local observation: {}".format(self.feature_num))

    def reset(self):
        # reset the indexes
        self.counter, self.cover_rate, self.collision = 0, 0, 0

        # choose directions randomly
        k = np.random.randint(4, size=1)[0]
        rot_mat = np.array([[np.cos(k/2*np.pi), -np.sin(k/2*np.pi)], [np.sin(k/2*np.pi), np.cos(k/2*np.pi)]])
        rot_mat1 = np.array([[np.cos((k+1)/2*np.pi), -np.sin((k+1)/2*np.pi)], [np.sin((k+1)/2*np.pi), np.cos((k+1)/2*np.pi)]])
        rot_mat2 = np.array([[np.cos((k+2)/2*np.pi), -np.sin((k+2)/2*np.pi)], [np.sin((k+2)/2*np.pi), np.cos((k+2)/2*np.pi)]])
        rot_mat3 = np.array([[np.cos((k+3)/2*np.pi), -np.sin((k+3)/2*np.pi)], [np.sin((k+3)/2*np.pi), np.cos((k+3)/2*np.pi)]])
        self.order = k

        # reset positions of targets
        tar_space = self.space + np.random.uniform(-0.2, 0.2, 2)  # [tar_row_space, tar_column_space]
        tar_pos = generate_pos_mat(self.tar_row_num, self.tar_column_num, tar_space[0], tar_space[1], self.ref_tar_pos)

        for i in range(self.n_targets+1):
            tar_pos[i] = np.dot(rot_mat, tar_pos[i])

        np.random.shuffle(tar_pos)
        self.tar_pos = copy.deepcopy(tar_pos)[0:48]

        # reset positions of robots
        pos_1 = np.zeros((int(0.25*self.n_agents), self.dim))
        pos_1[0] = np.random.uniform([-3.0, 4.5], [3.0, 6.5], self.dim)

        for i in range(1, int(0.25*self.n_agents)):
            coll = 1  # collision: coll = 1; safe: coll = 0.
            while coll == 1:
                coll = 0
                pos_1[i] = np.random.uniform([-3.0, 4.5], [3.0, 6.5], self.dim)
                for j in range(self.n_agents):
                    if j < i:
                        if np.linalg.norm(pos_1[i] - pos_1[j]) < (self.safe_dis + 0.2):
                            coll = 1
                            break

        pos_2 = np.zeros((int(0.25*self.n_agents), self.dim))
        pos_2[0] = np.random.uniform([-3.0, 4.5], [3.0, 6.5], self.dim)

        for i in range(1, int(0.25*self.n_agents)):
            coll = 1  # collision: coll = 1; safe: coll = 0.
            while coll == 1:
                coll = 0
                pos_2[i] = np.random.uniform([-3.0, 4.5], [3.0, 6.5], self.dim)
                for j in range(self.n_agents):
                    if j < i:
                        if np.linalg.norm(pos_2[i] - pos_2[j]) < (self.safe_dis + 0.2):
                            coll = 1
                            break

        pos_3 = np.zeros((int(0.25*self.n_agents), self.dim))
        pos_3[0] = np.random.uniform([-3.0, 4.5], [3.0, 6.5], self.dim)

        for i in range(1, int(0.25*self.n_agents)):
            coll = 1  # collision: coll = 1; safe: coll = 0.
            while coll == 1:
                coll = 0
                pos_3[i] = np.random.uniform([-3.0, 4.5], [3.0, 6.5], self.dim)
                for j in range(self.n_agents):
                    if j < i:
                        if np.linalg.norm(pos_3[i] - pos_3[j]) < (self.safe_dis + 0.2):
                            coll = 1
                            break

        pos_4 = np.zeros((int(0.25*self.n_agents), self.dim))
        pos_4[0] = np.random.uniform([-3.0, 4.5], [3.0, 6.5], self.dim)

        for i in range(1, int(0.25*self.n_agents)):
            coll = 1  # collision: coll = 1; safe: coll = 0.
            while coll == 1:
                coll = 0
                pos_4[i] = np.random.uniform([-3.0, 4.5], [3.0, 6.5], self.dim)
                for j in range(self.n_agents):
                    if j < i:
                        if np.linalg.norm(pos_4[i] - pos_4[j]) < (self.safe_dis + 0.2):
                            coll = 1
                            break

        for i in range(int(0.25*self.n_agents)):
            pos_1[i] = np.dot(rot_mat, pos_1[i])
            pos_2[i] = np.dot(rot_mat1, pos_2[i])
            pos_3[i] = np.dot(rot_mat2, pos_3[i])
            pos_4[i] = np.dot(rot_mat3, pos_4[i])

        np.random.shuffle(pos_1)  # reorder
        np.random.shuffle(pos_2)
        np.random.shuffle(pos_3)
        np.random.shuffle(pos_4)

        pos = np.concatenate((pos_1, pos_2, pos_3, pos_4), axis=0)

        self.global_pos = copy.deepcopy(pos)
        self.start_pos = copy.deepcopy(pos)

        # reset the adjacent matrix and neighbors
        self.adj_mat = np.array(kneighbors_graph(pos, self.degree, mode='connectivity', include_self=False).todense())

        return self._get_obs()

    def step(self, action):
        """
        update global states, observation and the communication topology
        :param action: global action of N robots, size=(self.n_agents*self.dim, )
        :return: observation at next time step and reward
        """
        self.counter = self.counter + 1

        action = np.reshape(action, (self.n_agents, self.dim))
        self.global_pos = self.global_pos + action*self.dt  # robot position

        obs_next = self._get_obs()  # calculate global feature
        reward = self._get_reward()

        self.adj_mat = np.array(kneighbors_graph(self.global_pos, self.degree, mode='connectivity', include_self=False).todense())

        return obs_next, reward

    def _get_obs(self):
        """
        calculate the global observation (feature)
        :return: a vector of observations, size=(self.n_agents*self.feature_num, )
        """
        # find the closest targets for all robots
        tar_nbrs = NearestNeighbors(n_neighbors=self.tar_num).fit(self.tar_pos)
        closest_tar_idx = tar_nbrs.kneighbors(self.global_pos, self.tar_num, return_distance=False)

        # find the closest neighbors for all robots
        nei_nbrs = NearestNeighbors(n_neighbors=self.rob_num).fit(self.global_pos)
        closest_nei_idx = nei_nbrs.kneighbors(self.global_pos, self.rob_num + 1, return_distance=False)

        # find the closest robots for all targets
        closest_rob_dist, closest_rob_idx = nei_nbrs.kneighbors(self.tar_pos, 2, return_distance=True)
        self.closest_rob_dist = closest_rob_dist
        self.closest_rob_idx = closest_rob_idx

        global_obs = np.zeros((self.n_agents, self.feature_num))
        for i in range(self.n_agents):
            tar_obs, nei_obs = [], []
            # obs of closest targets (based on Euclidean distance)
            for j in range(self.tar_num):
                idx = closest_tar_idx[i][j]
                delta_pos = self.tar_pos[idx] - self.global_pos[i]  # g_{i[j]} - p_i
                tar_obs.append(delta_pos)
                # find the closest robots for target idx
                tar_obs.append(self.tar_pos[idx] - self.global_pos[closest_rob_idx[idx][0]])

            # obs of closest neighbors (based on Euclidean distance)
            for k in range(1, self.rob_num + 1):
                idx = closest_nei_idx[i][k]
                delta_pos = self.global_pos[idx] - self.global_pos[i]  # p_{i[k]} - p_i
                nei_obs.append(delta_pos)

            global_obs[i] = np.concatenate(tar_obs + nei_obs)

        return global_obs.reshape(-1)

    def _get_reward(self):
        robot_dist = self.closest_rob_dist  # size=(self.n_targets, 2)

        # coverage reward
        reward_c, cover_num, bonus = 0, 0, 0
        reward_c = -np.sum(robot_dist[:, 0])  # distance between the landmark and its closest robot

        for i in range(self.n_targets):
            if robot_dist[i][0] < self.tar_r:
                cover_num = cover_num + 1
                bonus = bonus + 1.0

        self.cover_rate = cover_num/self.n_targets

        # collision reward
        collision_times = 0

        for i in range(self.n_targets):
            if robot_dist[i][1] < self.tar_r:
                collision_times = collision_times + 1

        self.collision = self.collision + collision_times

        # final reward
        self.cover_reward = reward_c + 1.0*bonus
        self.collision_reward = -5.0*collision_times
        reward = self.cover_reward + self.collision_reward

        return reward

    def render(self):
        """
        Render the environment with agents as circles in 2D space
        """
        tar_pos = self.tar_pos
        pos = self.global_pos

        if self.fig is None:
            plt.ion()  # dynamic figures
            fig = plt.figure(dpi=140)
            ax = fig.add_subplot(111)

            start_ax, = ax.plot(self.start_pos[:, 0], self.start_pos[:, 1], 'kx')  # black "x": start positions
            tar_ax, rob_ax, comm_ax = [], [], []
            for i in range(self.n_agents):
                rob_pos = self.rob_circle + pos[i]
                rob_ax_i, = ax.plot(rob_pos[:, 0], rob_pos[:, 1], 'b-', linewidth=1.5)  # blue circle: robot
                rob_ax.append(rob_ax_i)

                for j in range(self.n_agents):
                    if j == i: continue
                    if self.adj_mat[i][j] > 0:
                        comm = np.array([pos[i], pos[j]])
                        comm_ax_ij, = ax.plot(comm[:, 0], comm[:, 1], 'g--', linewidth=0.7)  # communication link
                        comm_ax.append(comm_ax_ij)

            for k in range(self.n_targets):
                temp_pos = self.tar_circle + tar_pos[k]
                tar_ax_k, = ax.plot(temp_pos[:, 0], temp_pos[:, 1], 'r-', linewidth=1.0)  # red circle: target
                tar_ax.append(tar_ax_k)

            plt.title('Cooperative Coverage')
            plt.axis('equal')
            self.fig = fig
            self.start_ax = start_ax
            self.tar_ax = tar_ax
            self.rob_ax = rob_ax
            self.comm_ax = comm_ax

        # update positions
        if self.counter == 1:
            self.start_ax.set_xdata(self.start_pos[:, 0])  # update the start positions
            self.start_ax.set_ydata(self.start_pos[:, 1])

            for i in range(self.n_targets):
                temp_pos = self.tar_circle + tar_pos[i]
                self.tar_ax[i].set_xdata(temp_pos[:, 0])
                self.tar_ax[i].set_ydata(temp_pos[:, 1])

        comm_num = 0
        for i in range(self.n_agents):
            rob_pos = self.rob_circle + pos[i]
            self.rob_ax[i].set_xdata(rob_pos[:, 0])
            self.rob_ax[i].set_ydata(rob_pos[:, 1])

            for j in range(self.n_agents):
                if j == i: continue
                if self.adj_mat[i][j] > 0:
                    comm = np.array([pos[i], pos[j]])
                    self.comm_ax[comm_num].set_xdata(comm[:, 0])
                    self.comm_ax[comm_num].set_ydata(comm[:, 1])
                    comm_num = comm_num + 1

        plt.xlim(-8.5, 8.5)
        plt.ylim(-6.5, 6.5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_circle(self, radius):
        """
        :return: an array with the size=(n_circle,dim)
        """
        n_circle = 20

        thetas = np.linspace(-np.pi, np.pi, n_circle)
        circle_pos = np.zeros((n_circle, self.dim))
        for i in range(n_circle):
            circle_pos[i] = radius*np.array([np.cos(thetas[i]), np.sin(thetas[i])])

        return circle_pos

    def close(self):
        pass


def generate_pos_mat(row_num, column_num, row_space, column_space, ref_pos):
    """
    :param row_num: number of rows
    :param column_num: number of columns
    :param row_space: distance between two neighbouring rows
    :param column_space: distance between two neighbouring columns
    :param ref_pos: center of the position matrix
    :return: an array of positions, size=(row_num*column_num, dim)
    """
    x = np.arange(column_num)*column_space - 0.5*(column_num - 1)*column_space + ref_pos[0]
    y = np.arange(row_num)*row_space - 0.5*(row_num - 1)*row_space + ref_pos[1]

    total_pos = np.zeros((row_num*column_num, 2))
    count = 0
    for i in range(column_num):
        for j in range(row_num):
            total_pos[count] = np.array([x[i], y[j]], dtype=float)
            count = count + 1

    return total_pos

