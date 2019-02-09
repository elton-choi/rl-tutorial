#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gym
from time import sleep

if __name__ == '__main__':
    env = gym.make("FrozenLake8x8-v0")
    env.reset()

    print(env.action_space)
    for _ in range(4):
        env.render()
        env.step(env.action_space.sample())
        sleep(1)
        