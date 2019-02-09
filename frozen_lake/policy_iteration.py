#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import wrappers
from time import sleep

class PolicyIteration:
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.pi = np.random.choice(env.nA, env.nS)
        self.v = np.zeros(env.nS)
        self.gamma = gamma

    def policy_evaluation(self):
        eps = 1e-10
        while True:
            v_pre = np.copy(self.v)

            for s in range(self.env.nS):
                a = self.pi[s]
                self.v[s] = sum([p * (r + self.gamma * v_pre[s_]) for p, s_, r, _ in self.env.P[s][a]])
            if (np.sum(np.fabs(v_pre - self.v)) <= eps):
                break
        
        return self.v

    def policy_improvement(self):
        pi = np.zeros(self.env.nS)
        for s in range(self.env.nS):
            q_sa = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                q_sa[a] = sum([p * (r + self.gamma * self.v[s_]) for p, s_, r, _ in self.env.P[s][a]])
            
            pi[s] = np.argmax(q_sa)
        
        return pi

    def iterate(self):
        n = 200000
        eps = 1e-20
        for i in range(n):
            self.policy_evaluation()
            pi_new = self.policy_improvement()

            if (np.all(self.pi == pi_new)):
                print('Policy-Iteration converged at step %d.' %(i+1))
                break
            
            self.pi = pi_new

        return self.pi

    def run_episode(self, render = False, key = False, wait = 0.0):
        obs = self.env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                self.env.render()
                if key:
                    temp_key = input('')
                elif wait > 0.0:
                    sleep(wait)

            obs, reward, done , _ = self.env.step(int(self.pi[obs]))
            total_reward += (self.gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward

if __name__ == '__main__':
    env = gym.make("FrozenLake8x8-v0")
    env = env.unwrapped
    gamma = 0.8
    policy_iteration = PolicyIteration(env, gamma)

    # random policy
    k = input('run 1 episode with ramdom policy?')
    score = policy_iteration.run_episode(render=True, key=False, wait=0.2)
    print('score(random policy) = ', score)

    # policy iteration
    policy_iteration.iterate()

    # 1 episode
    k = input('run 1 episode with policy iteration?')
    score = policy_iteration.run_episode(render=True, key=False, wait=0.2)
    print('score(policy iteration, 1 episode) = ', score)

    # multiple episodes
    k = input('run 100 episodes with policy iteration?')
    scores = [policy_iteration.run_episode() for _ in range(100)]
    score = np.mean(scores)
    print('Average score(policy iteration, 100 episodes) = ', score)