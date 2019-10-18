import numpy as np
from preprocess import preprocess
import Gradient_Descent as gd


# loads the data
train = preprocess("data/PA1_train.csv")
val = preprocess("data/PA1_dev.csv", norm=train.norm)
test = preprocess("data/PA1_test.csv", test=True, norm=train.norm)

# You can get the unnormalized X with:
Print("training X:")
print(train.X)
# or normaized as follows
print("normalized training X")
print(train.X_norm)
# same with y:
print("training y:")
print(train.y)
print("normalized training y:")
print(train.y_norm)

# validation and test work the same:
print("normalized validation X:")
print(val.X_norm)
print("normalized test X:")
print(test.X_norm)

