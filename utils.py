import os
import torch
import random
import numpy as np


def set_seed(seed_num: int):
  '''seed fix'''
  random.seed(seed_num)
  os.environ['PYTHONHASHSEED'] = str(seed_num)
  np.random.seed(seed_num)
  torch.manual_seed(seed_num)
  torch.cuda.manual_seed(seed_num)
  torch.cuda.manual_seed_all(seed_num)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False