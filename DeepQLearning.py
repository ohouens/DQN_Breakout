#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import random
import pickle
from skimage import transform, io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque
from DQN import DQN
from Memory import Memory
from os import path
from tensorflow import keras

class DeepQLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, batchSize=32, stackSize=4, filename="model"):
        self.env = env
        self.filename = filename
        self.model_name = self.filename+"_model"
        self.memory_name = self.filename+"_memory"
        self.alpha = alpha
        self.gamma = gamma
        self.batchSize = batchSize
        self.stackSize = stackSize
        self.actionSize = env.action_space.n
        self.stateSize = [84,84,4]
        self.possibleActions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
        self.DQNetwork = DQN((84,84,4), self.actionSize, self.alpha, self.batchSize)
        if path.exists(self.model_name):
            self.DQNetwork.model = keras.models.load_model(self.model_name, custom_objects={'my_loss_fn': self.DQNetwork.my_loss_fn})
            print("model loaded")
        else:
            self.DQNetwork.my_compile()
        
    def preprocessing(self, frame, show):
        if show:
            io.imshow(frame)
            plt.show()
        grayScale = rgb2gray(frame)
        normalizedFrame = grayScale/255.0
        result = transform.resize(normalizedFrame, [110,84])
        result = result[19:-7,:]
        if show:
            io.imshow(result)
            plt.show()
        return result
    
    def stackFrames(self, stackedFrames, state, isNewEpisode, show=False):
        frame = self.preprocessing(state, show)
        if isNewEpisode:
            stackedFrames = deque([np.zeros((84,84), dtype=np.int) for i in range(self.stackSize)], maxlen=4)
            stackedFrames.append(frame)
            stackedFrames.append(frame)
            stackedFrames.append(frame)
            stackedFrames.append(frame)
        else:
            stackedFrames.append(frame)
        stackedState = np.stack(stackedFrames, axis=2)
        return stackedState, stackedFrames
        
    def train(self, totalEpisodes=100, maxSteps=100, memorySize=1000, maxEpsilon=1, minEpsilon=0.1, lastGreedyStep=1000, episodeRender=False):
        #initiate memory
        stackedFrames = deque([np.zeros((84,84), dtype=np.int) for i in range(self.stackSize)], maxlen=4)
        if path.exists(self.memory_name):
            with open(self.memory_name, 'rb') as memory_file:
                memory = pickle.load(memory_file)
        else:
            memory = Memory(maxSize=self.batchSize)
        for i in range(self.batchSize):
            if i == 0:
                state = env.reset()
                state, stackedFrames = self.stackFrames(stackedFrames, state, True, True)
            
            choice = random.randint(1,len(self.possibleActions))-1
            action = self.possibleActions[choice]
            nextState, reward, done, _ = env.step(np.argmax(action))
            
            nextState, stackedFrames = self.stackFrames(stackedFrames, nextState, False)
            
            if done:
                nextState = np.zeros(state.shape)
                memory.add((state, action, reward, nextState, done))
                state = env.reset()
                state, stackedFrames = self.stackFrames(stackedFrames, state, True)
            else:
                memory.add((state, action, reward, nextState, done))
                state = nextState
        
        
        #training loop
        exploreProbability = maxEpsilon
        decayRate = (maxEpsilon-minEpsilon)/lastGreedyStep
        rewardList = []
        loss = 0
        for episode in range(totalEpisodes):
            step = 0
            episodeRewards = []
            state = env.reset()
            state, stackedFrames = self.stackFrames(stackedFrames, state, True)
            while step < maxSteps:
                #if(step == 100):
                #    for frame in stackedFrames:
                #        io.imshow(frame)
                #        plt.show()
                step += 1
                action = self.predictAction(exploreProbability, state)
                nextState, reward, done, _ = env.step(np.argmax(action))
                
                exploreProbability -= decayRate
                exploreProbability = max(exploreProbability, minEpsilon)
                
                if episodeRender:
                    env.render()
                    
                episodeRewards.append(reward)
                if done:
                    nextState = np.zeros((84, 84), dtype=np.int)
                    nextState, stackedFrames = self.stackFrames(stackedFrames, nextState, False)
                    totalReward = np.sum(episodeRewards)
                    print('Episode: {}\n'.format(episode),
                          'Total reward: {}\n'.format(totalReward),
                          'Explore P: {:.4f}\n'.format(exploreProbability),
                          'Training Loss: {:.4f}\n'.format(loss),
                          'Steps needed: {}\n'.format(step))
                    step = maxSteps
                    rewardList.append((episode, totalReward))
                    memory.add((state, action, reward, nextState, done))
                else:
                    nextState, stackedFrames = self.stackFrames(stackedFrames, nextState, False)
                    memory.add((state, action, reward, nextState, done))
                    state = nextState
                    
                batch = memory.sample(self.batchSize)
                statesMb = np.array([each[0] for each in batch], ndmin=3)
                actionsMb = np.array([each[1] for each in batch])
                rewardsMb = np.array([each[2] for each in batch])
                nextStatesMb = np.array([each[3] for each in batch], ndmin=3)
                donesMb = np.array([each[4] for each in batch])
                targetQsBatch = []
                #print("next states mb:", nextStatesMb.shape)
                #print("dqn states size:", self.DQNetwork.stateSize)
                #self.DQNetwork.model.summary()
                QsNextState = self.DQNetwork.model(nextStatesMb)
                for i in range(len(batch)):
                    terminal = donesMb[i]
                    if terminal:
                        targetQsBatch.append(rewardsMb[i])
                    else:
                        target = rewardsMb[i] + self.gamma * np.max(QsNextState[i])
                        targetQsBatch.append(target)
                targetsMb = np.array([each for each in targetQsBatch])
                
                loss = self.DQNetwork.batch_loss(statesMb, targetsMb, actionsMb)
                
            if episode % 5==0:
                self.DQNetwork.model.save(self.model_name)
                with open(self.memory_name, 'wb') as memory_file:
                    pickle.dump(memory, memory_file)
                print("Model saved")
    
    def predictAction(self, exploreProbability, state):
        expExpTradeoff = np.random.rand()
        if(exploreProbability > expExpTradeoff):
            choice = random.randint(0, len(self.possibleActions)-1)
            action = self.possibleActions[choice]
        else:
            Qs = self.DQNetwork.model(state.reshape((1, *state.shape)))
            choice = np.argmax(Qs)
            action = self.possibleActions[choice]
        return action
    
    def play(self, state):
        Qs = self.DQNetwork.model(state.reshape((1, *state.shape)))
        return np.argmax(Qs)

if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    print("action space:", env.action_space.n)
    print("observation space", list(env.observation_space.shape))
    print(env.observation_space)
    state = env.reset()
    done = False
    learning = DeepQLearning(env, alpha=0.00025, batchSize=32, filename="result/day_1")
    learning.train(totalEpisodes=10000, maxSteps=10000, lastGreedyStep=1000000, memorySize=1000000, episodeRender=True)
    #while not done:
    #    state, reward, done, _ = env.step(learning.play(learning.preprocessing(state)))
    #    env.render()
    env.close()    