import numpy as np
import gym
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt

#
# This class implements the Q Learning algorithm.
# We can use this implementation to solve Toy text environments from Gym project. 
#

class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        (x,y,z) = env.observation_space
        self.q_table = np.zeros([(x.n*y.n*z.n), env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def stateNumber(self,state):
        (x,y,z) = state
        y = y * 32
        z = z * 352
        return x+y+z

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explore action space
        return np.argmax(self.q_table[state]) # Exploit learned values

    def train(self, filename, plotFile):
        actions_per_episode = []
        penalties = 0
        draw = 0
        win = 0
        lose = 0
        for i in range(1, self.episodes+1):
            state = self.env.reset()
            state=self.stateNumber(state)
            reward = 0
            done = False
            actions = 0
            
            

            while not done:                
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.stateNumber(next_state) 
        
                # Adjust Q value for current state
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[state, action] = new_value
                
                state = next_state
                actions += 1
            
            if reward == -1:
                penalties+=1
                lose+=1
            elif reward == 0:
                draw+=1
            elif reward==1:
                win+=1
                
                

            if i % 100 == 0:                
                actions_per_episode.append(penalties/100)
                penalties = 0
                #sys.stdout.write("Episodes: " + str(i) +'\r')
                #sys.stdout.flush()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec        
        
        draw=draw/self.episodes
        win=win/self.episodes
        lose=lose/self.episodes

        with open("data/Output.txt", "a") as text_file:
            print(f"Win rate: {win}", file=text_file)
            print(f"Lose rate: {lose}", file=text_file)
            print(f"Draw rate: {draw}", file=text_file)
            print(f"Alpha: {self.alpha}", file=text_file)
            print(f"Gamma: {self.gamma}", file=text_file)
            print(f"Episodes: {self.episodes}", file=text_file)

        savetxt(filename, self.q_table, delimiter=',')
        if (plotFile is not None): self.plotactions(plotFile, actions_per_episode)
        return self.q_table

    def plotactions(self, plotFile, actions_per_episode):
        plt.plot(actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Penalties')
        plt.title('# Penalties vs Episodes')
        plt.savefig(plotFile+".jpg")     
        plt.close()

        size = len(actions_per_episode)
        start = (int)(size - (size * 0.1))
        plt.plot(actions_per_episode[start:])
        plt.xlabel('Episodes')
        plt.ylabel('# Penalties')
        plt.title('# Penalties vs Episodes (Last episodes)')
        plt.savefig(plotFile+"_last.jpg")     
        plt.close()
