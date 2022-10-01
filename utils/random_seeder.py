import numpy as np
import torch as tc
import random

def set_random_seed(seed):		
	random.seed(seed)
	np.random.seed(seed)
	tc.manual_seed(seed)
	tc.cuda.manual_seed(seed)
	tc.cuda.manual_seed_all(seed)


def set_cudnn_backends():
	set_random_seed(0)
	tc.backends.cudnn.deterministic = True
	tc.backends.cudnn.benchmark = False
	tc.backends.cudnn.enabled = False   # 禁用非确定性算法