#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
class Memory():
    def __init__(self, maxSize):
        self.buffer = deque(maxlen=maxSize)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batchSize):
        bufferSize = len(self.buffer)
        index = np.random.choice(np.arange(bufferSize), size=batchSize, replace=False)
        return [self.buffer[i] for i in index]
