# -*- coding: utf-8 -*-
"""
@author: Manny Ko

"""
import random
import numpy as np
#import torch


def initSeeds(seed=1):
	print(f"initSeeds({seed})")
	random.seed(seed)
#	torch.manual_seed(seed) 	#turn on this 2 lines when torch is being used
#	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
