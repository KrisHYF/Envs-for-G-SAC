"""
Flocking Navigation Environment (Version I)
Problem Formulation: there are N mobile robots as the followers and a virtual robot as the leader, the average position
                     of followers should track the position of the leader, and for any pair of robots i and j, the
                     relative distance should satisfy abs(dist_ij-d) ≤ △d.

Author: Yifan Hu, SEU
Date: 08/07/2022
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors


class FlockingNavigation:
    def __init__(self, args):
        # individual robot parameters
        self.dt = 0.1  # sampling time interval
        self.dim = 2  # 2-D robot
        self.r = 0.1  # robot radius (default: 0.1)
        self.desire_dis = 0.5  # desired distance (default: 0.5)

        # problem parameters
        self.n_agents = args.n_agents
        self.degree = args.degree  # number of communication neighbors
        self.rob_num = args.rob_num  # number of observed closest robots
        self.feature_num = self.dim*(self.rob_num + 2)  # {p_i - p_j}, p_i - p_tar, v_tar

        self.global_pos = np.zeros((self.n_agents, self.dim))  # global positions of N robots
        self.ave_pos = np.zeros((self.dim, ))  # average position of N robots
        self.input = None  # control input
        self.adj_mat = np.zeros((self.n_agents, self.n_agents))  # adjacent matrix

        print("Number of agents: {}".format(self.n_agents))
        print("Dim of global feature: {}".format(self.n_agents*self.feature_num))

        self.tar_pos = np.zeros((self.dim, ))  # target position
        self.tar_u = np.zeros((self.dim, ))  # target velocity

        # task indexes
        self.counter = 0
        self.closest_idx = None
        self.total_flock_num = self.rob_num * self.n_agents
        self.track_rate = 0  # here, it represents the distance between the flocking center and the leader's position
        self.flock_rate = 0  # flocking rate
        self.track_reward = 0
        self.flock_reward = 0

        # figure parameters
        self.fig = None
        self.start_ax = None
        self.tar_ax = None  # draw target position
        self.rob_ax = None  # draw robot position
        self.comm_ax = None  # draw communication link
        self.ave_ax = None  # draw average position

        self.start_pos = None  # record the start positions
        self.rob_circle = self.draw_circle(self.r, True)
        self.tar_circle = self.draw_circle(0.3*self.r, True)
        self.ave_circle = self.draw_circle(0.3*self.r, True)

    def reset(self):
        # reset the indexes
        self.counter, self.track_rate, self.flock_rate = 0, 0, 0

        # reset the states of the target
        self.tar_pos = np.random.uniform([2.0, -0.1], [2.5, 0.1], self.dim)
        self.tar_u = np.zeros((self.dim, ))
        self.tar_u[0] = np.random.uniform(0.6, 0.8)  # TODO - more complex movements

        # reset the states of robots
        pos = np.zeros((self.n_agents, self.dim))
        pos[0] = np.random.uniform([-1.5, -2.0], [1.0, 2.0], self.dim)  # [-1.0, -2.0], [1.0, 2.0] for 10 robots

        for i in range(1, self.n_agents):
            coll = 1  # collision: coll = 1; safe: coll = 0.
            while coll == 1:
                coll = 0
                pos[i] = np.random.uniform([-1.5, -2.0], [1.0, 2.0], self.dim)
                for j in range(self.n_agents):
                    if j < i:
                        if np.linalg.norm(pos[i] - pos[j]) < (2*self.r + 0.2):
                            coll = 1
                            break

        np.random.shuffle(pos)

        self.global_pos = copy.deepcopy(pos)
        self.start_pos = copy.deepcopy(pos)

        # reset the adjacent matrix and neighbors
        self.adj_mat = np.array(kneighbors_graph(pos, self.degree, mode='connectivity', include_self=False).todense())

        return self._get_obs()

    def step(self, action):
        """
        update global states, observation and the communication topology
        :param action: size=(self.n_agents*self.dim, )
        :return: observation at next time step and reward
        """
        # TODO - adjust target's velocity in this function
        self.counter = self.counter + 1

        action = np.reshape(action, (self.n_agents, self.dim))
        self.global_pos = self.global_pos + action*self.dt  # robot position
        self.tar_pos = self.tar_pos + self.tar_u*self.dt  # target position
        self.input = action

        obs_next = self._get_obs()
        reward = self._get_reward()

        self.adj_mat = np.array(kneighbors_graph(self.global_pos, self.degree, mode='connectivity', include_self=False).todense())

        return obs_next, reward

    def _get_obs(self):
        """
        calculate the global observation (feature) and the average team position
        :return: a vector of observations, size=(self.n_agents*self.feature_num, )
        """
        self.ave_pos = np.mean(self.global_pos, axis=0)

        # find the closest neighboring robots
        nei_nbrs = NearestNeighbors(n_neighbors=self.rob_num).fit(self.global_pos)
        closest_idx = nei_nbrs.kneighbors(self.global_pos, self.rob_num + 1, return_distance=False)
        self.closest_idx = closest_idx  # size=(self.n_agents, self.rob_num+1)

        # calculate global observation
        pos = self.global_pos
        global_obs = np.zeros((self.n_agents, self.feature_num))

        for i in range(self.n_agents):
            delta_pos = self.tar_pos - pos[i]  # relative position
            target_state = [delta_pos] + [self.tar_u]  # state of target

            neighbor_state = []  # relative positions w.r.t. neighbors
            for j in range(1, self.rob_num + 1):
                neighbor_state.append(pos[closest_idx[i][j]] - pos[i])

            global_obs[i] = np.concatenate(target_state + neighbor_state)

        return global_obs.reshape(-1)

    def _get_reward(self):
        pos = self.global_pos
        # tracking reward
        rel_dist = np.linalg.norm(self.tar_pos - self.ave_pos)
        reward_t = -rel_dist
        if rel_dist < 0.05:
            bonus_t = 1
        else:
            bonus_t = 0

        reward_tracking = 2.0*reward_t + 0.5*bonus_t
        self.track_rate = rel_dist  # distance between flocking center and leader's position

        # flocking reward
        reward_f, flock_num = 0, 0
        for i in range(self.n_agents):
            for j in range(1, self.rob_num + 1):
                delta_pos = pos[i] - pos[self.closest_idx[i][j]]
                flocking_err = np.abs(np.linalg.norm(delta_pos) - self.desire_dis)
                reward_f = reward_f - flocking_err
                if flocking_err < 0.05:
                    flock_num = flock_num + 1

        reward_flocking = 0.30*(reward_f + flock_num)
        self.flock_rate = flock_num/self.total_flock_num

        # control input reward (not used in the current version)
        reward_input = 0
        for i in range(self.n_agents):
            reward_input -= 0.01*np.linalg.norm(self.input[i])

        # final reward
        self.track_reward = reward_tracking
        self.flock_reward = reward_flocking

        reward = self.track_reward + self.flock_reward + 0.0*reward_input

        return reward

    def render(self):
        """
        Render the environment with agents as circles in 2D space
        """
        pos = self.global_pos

        if self.fig is None:
            plt.ion()  # dynamic figures
            fig = plt.figure(dpi=140)
            ax = fig.add_subplot(111)

            start_ax, = ax.plot(self.start_pos[:, 0], self.start_pos[:, 1], 'kx')  # black "x": start positions
            tar_pos = self.tar_circle + self.tar_pos
            tar_ax, = ax.plot(tar_pos[:, 0], tar_pos[:, 1], 'k', linewidth=2)  # leader
            ave_pos = self.ave_circle + self.ave_pos
            ave_ax, = ax.plot(ave_pos[:, 0], ave_pos[:, 1], 'r', linewidth=2)  # average team position
            rob_ax, comm_ax = [], []
            for i in range(self.n_agents):
                rob_pos = self.rob_circle + pos[i]
                rob_ax_i, = ax.plot(rob_pos[:, 0], rob_pos[:, 1], 'b-', linewidth=1.5)  # robot
                rob_ax.append(rob_ax_i)

                for j in range(self.n_agents):
                    if j == i: continue
                    if self.adj_mat[i][j] > 0:
                        comm = np.array([pos[i], pos[j]])
                        comm_ax_ij, = ax.plot(comm[:, 0], comm[:, 1], 'g-', linewidth=0.7)  # communication link
                        comm_ax.append(comm_ax_ij)

            plt.title('Flocking Navigation')
            plt.axis('equal')
            self.fig = fig
            self.start_ax = start_ax
            self.tar_ax = tar_ax
            self.ave_ax = ave_ax
            self.rob_ax = rob_ax
            self.comm_ax = comm_ax

        # update positions
        if self.counter == 1:
            self.start_ax.set_xdata(self.start_pos[:, 0])
            self.start_ax.set_ydata(self.start_pos[:, 1])

        tar_pos = self.tar_circle + self.tar_pos
        self.tar_ax.set_xdata(tar_pos[:, 0])
        self.tar_ax.set_ydata(tar_pos[:, 1])

        ave_pos = self.ave_circle + self.ave_pos
        self.ave_ax.set_xdata(ave_pos[:, 0])
        self.ave_ax.set_ydata(ave_pos[:, 1])

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

        plt.xlim(self.tar_pos[0] - 5, self.tar_pos[0] + 5)
        plt.ylim(self.tar_pos[1] - 5, self.tar_pos[1] + 5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_circle(self, radius, agent=True):
        """
        :return: an array with the size=(n_circle,2)
        """
        if agent:
            n_circle = 20  # agent
        else:
            n_circle = 50  # target

        thetas = np.linspace(-np.pi, np.pi, n_circle)
        circle_pos = np.zeros((n_circle, self.dim))
        for i in range(n_circle):
            circle_pos[i] = radius*np.array([np.cos(thetas[i]), np.sin(thetas[i])])

        return circle_pos

    def close(self):
        pass
