#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import wrappers
from time import sleep

class ValueIteration:
    def __init__(self, env, gamma=0.8):
        self.env = env
        self.pi = np.random.choice(env.nA, env.nS)
        self.v = np.zeros(env.nS)
        self.gamma = gamma

    def value_iteration(self):
        v = self.v.copy()
        for s in range(self.env.nS):
            q_sa = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                q_sa[a] = sum([p * (r + self.gamma * self.v[s_]) for p, s_, r, _ in self.env.P[s][a]])
            
            v[s] = np.max(q_sa)          
        
        return v
    
    def get_optimal_policy(self):
        for s in range(self.env.nS):
            q_sa = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                q_sa[a] = sum([p * (r + self.gamma * self.v[s_]) for p, s_, r, _ in self.env.P[s][a]])
            
            self.pi[s] = np.argmax(q_sa)        
        
        return self.pi

    def iterate(self):
        n = 200000
        eps = 1e-20
        for i in range(n):
            v_new = self.value_iteration()

            if (np.all(self.v == v_new)):
                print('Value-Iteration converged at step %d.' %(i+1))
                break
            
            self.v = v_new
        
        return self.get_optimal_policy()

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
    value_iteration = ValueIteration(env, gamma)

    # random policy
    k = input('run 1 episode with ramdom policy?')
    score = value_iteration.run_episode(True, False, 0.1)
    print('score(random policy) = ', score)

    # policy iteration
    value_iteration.iterate()

    # 1 episode
    k = input('run 1 episode with value iteration?')
    score = value_iteration.run_episode(True, False, 0.1)
    print('score(value iteration, 1 episode) = ', score)

    # multiple episodes
    k = input('run 100 episodes with policy iteration?')
    scores = [value_iteration.run_episode(False) for _ in range(100)]
    score = np.mean(scores)
    print('Average score(value iteration, 100 episodes) = ', score)