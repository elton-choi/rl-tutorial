#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gym
from time import sleep
import math

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.8, epsilon=0.1):
        self.env = env     
        self.alpha = alpha   
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n
        self.q_table = np.zeros((self.n_state, self.n_action))
        self.S = np.arange(self.n_state)
        self.A = np.arange(self.n_action)
        self.str_a = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        
    def explore(self, s):
        # epsilon-greedy
        if np.random.rand() < self.epsilon:
            a = np.random.choice(self.A)
        else:   # greedy: 1-e
            a = np.argmax(self.q_table[s])
        
        # print('s=%d, a=%s\n' % (s, self.str_a[a]))    

        return a

    def learn(self, s, a, r, s_):
        # print('\nbefore update')
        # self.print_q_table()
        # print('\ns=%d, a=%d, r=%.2f, s_=%d'%(s, a, r, s_))    
        self.q_table[s][a] = (1-self.alpha)*self.q_table[s][a] + self.alpha * ( r + self.gamma * np.max(self.q_table[s_]) - self.q_table[s][a] )
        # print('\nafter q table')
        # self.print_q_table()
        # temp = input('Go next?')
        # print('----\n')
        
    def print_q_table(self):
        
        print('q_table')
        print('action = LEFT, DOWN, RIGHT, UP')
        for state in range(self.n_state):
            for action in range(self.n_action):
                if action == self.n_action - 1:
                    print('%.1f ' % self.q_table[state][action], end='')
                else:
                    print('%.1f, ' % self.q_table[state][action], end='')
            
            print('| ', end='')
            n_square = np.sqrt(self.n_state)
            if state % n_square == n_square-1:
                print('')

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

            a = self.explore(obs)
            next_obs, reward, done , _ = self.env.step(a)
            
            # print('obs = %d, a = %d, next_obs = %d\n'%(obs, a, next_obs))
            # temp_key = input('')
            
            obs = next_obs
            # total_reward += (self.gamma ** step_idx * reward)
            total_reward += (reward)
            step_idx += 1
            if done:
                break
        return (step_idx, reward, total_reward)

if __name__ == '__main__':
    # env = gym.make("FrozenLake-v0")
    env = gym.make("FrozenLake8x8-v0")
    agent = QLearning(env, alpha=0.8, gamma=0.9, epsilon=0.2)

    # before learning
    step_idx, reward, total_reward = agent.run_episode(render=False)
    print('Run episode before learning')
    print('Total reward is %.3f\n\n' % total_reward)

    # q learning
    for i in range(10000):
        obs = env.reset()
        t=1
        t_list = []
        action_list = []
        reward_list = []
        obs_list = []
        while True:
            # env.render()

            action = agent.explore(obs) # epsilon greedy
            next_obs, reward, done, info = env.step(action)
        
            obs_list.append(obs)
            action_list.append(action)

            if done:
                if reward == 0:
                    reward = -1

                reward_list.append(reward)                    
                agent.learn(obs, action, reward, next_obs)         # q-learning
                t_list.append(t)
                if reward == 1:
                    print("%dth Episode finished after %d timesteps with average reward %.3f" % (i+1, t, sum(reward_list)/t))
                    print("Observation list = {}".format(obs_list))
                    print("Action list = {}".format(action_list))
                break
            else:
                if reward == 0:
                    reward = -0.1
                
                reward_list.append(reward)
                agent.learn(obs, action, reward, next_obs)         # q-learning
            
            obs = next_obs

            t=t+1
    
    
    agent.print_q_table()
    print("Shortest time step to reach a goal is %d"%min(t_list))
    input('Next?')
    reward_list = []
    step_list = []
    n_episode = 1000
    agent.epsilon = 0.0
    for i in range(n_episode):
        (step_idx, reward, total_reward) = agent.run_episode(render=True)
        reward_list.append(total_reward)
        if reward == 1:
            step_list.append(step_idx)
            print("%dth episode finished at %d step with total reward %.3f" % (i, step_idx, total_reward))
            input('Next?')
    
    print('\nRun episodes(%d) after learning' % n_episode)
    print('Total reward average is %.3f' % (sum(reward_list)/n_episode))
    print('Shortest time step is %d' % min(step_list))