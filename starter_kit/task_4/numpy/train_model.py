'''
a dummy script to "train" a model and save the "trained model"
(just saved the model outputs of an simple model for local testing) 
'''

import numpy as np
#from irt import MyModel
from irt_best import MyModel

model = MyModel()
model.train_model()