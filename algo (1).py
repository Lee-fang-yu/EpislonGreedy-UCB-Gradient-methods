# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:08:17 2019

@author: user
"""

import numpy as np

class EpislonGreedy(object):
    
    def __init__(self, NumofBandits=10, epislon=0.1):
        assert (0. <= epislon <= 1.0)
        self._epislon = epislon
        self._nb = NumofBandits
        self._Q=np.zeros(self._nb,dtype=float)
        self._action_N= np.zeros(self._nb, dtype=int)
        
    def update(self, action, immi_reward):
        self._action_N[action] += 1
        self._Q[action] += (1/self._action_N[action])*(immi_reward - self._Q[action])
        
    def act(self, t):
        if(np.random.random() > self._epislon):
             # greedy
            a = np.argmax(self._Q)
            return a
        else:
            # exploration
            a = np.random.randint(0, self._nb)
            return a
            

class UCB(object):
    
    def __init__(self, NumofBandits=10, c=2):
        self._nb = NumofBandits
        self._c = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)
        
    def update(self, action, immi_reward):
        self._action_N[action] += 1
        self._Q[action] += (1./self._action_N[action]) * (immi_reward - self._Q[action])
        
    def act(self, t):
        if(t < self._nb):
            a = np.random.randint(self._nb)
            if self._action_N[a] == 0:
                return a
            else: 
                return np.argmin(self._action_N)
        else:
            return np.argmax(self._Q + self._c * np.sqrt(np.log(t) / self._action_N))
        
class Gradient(object):
    
    def __init__(self, NumofBandits=10, alpha=0.1):
        self._nb = NumofBandits
        self._H = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)
        self._pi = np.zeros(self._nb, dtype=float)
        self._t = 0
        self._avg_reward = 0
        self._alpha = alpha
        for i in range(self._nb):
            self._pi[i]=1/self._nb
    def update(self, action, immi_reward):
        self._avg_reward = self._avg_reward +(immi_reward-self._avg_reward)/self._t
        for i in range(self._nb):
            if (i==action):
                self._H[i]=self._H[i]+self._alpha*(immi_reward-self._avg_reward)*(1-self._pi[i])
            else:
                self._H[i]=self._H[i]-self._alpha*(immi_reward-self._avg_reward)*self._pi[i]
        sum_h = 0
        for i in range(self._nb):
            sum_h += np.exp(self._H[i])
        for i in range(self._nb):
            self._pi[i]=np.exp(self._H[i])/sum_h
            
                
    def act(self, t):
        self._t = t+1
        return np.random.choice(self._nb,p= self._pi)