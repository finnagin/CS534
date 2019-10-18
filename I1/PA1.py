import numpy as np
from preprocess import preprocess
import Gradient_Descent as gd


train = preprocess("data/PA1_train.csv")
val = preprocess("data/PA1_dev.csv", norm=train.norm)
test = preprocess("data/PA1_test.csv", test=True, norm=train.norm)

train.get_stats()
