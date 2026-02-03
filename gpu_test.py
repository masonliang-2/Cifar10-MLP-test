from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))