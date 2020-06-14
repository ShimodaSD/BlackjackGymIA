import gym
import numpy as np
import matplotlib.pyplot as plt
from QLearning import QLearning
from numpy import loadtxt

def stateNumber(state):
        (x,y,z) = state
        y = y * 32
        z = z * 352
        return x+y+z

env = gym.make('Blackjack-v0')
for i in [0.01]:
    for g in [0.000001,0.00001,0.0001,0.001,0.01]:
        for epi in [600000,700000,800000]:
            qlearn = QLearning(env, alpha=i, gamma=g, epsilon=0.9,epsilon_min=0.01, epsilon_dec=0.99, episodes=epi)
            q_table = qlearn.train('data/q-table-blackjack.csv', 'results/blackjack')
#q_table = loadtxt('data/q-table-blackjack.csv', delimiter=',')

#state= env.reset()
#print(state) 
#state = stateNumber(state)
#done = False
#
#
#while not done:
#    action = np.argmax(q_table[state])
#    state, reward, done, info = env.step(action)
#    print(action)
#    print(state)
#    state = stateNumber(state)
#    
#print(env.player)
#print(reward)
#print(env.dealer)