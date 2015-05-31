

import pandas as pd
import numpy as np
import csv as csv


#load the train data
train_df = pd.read_csv('../data/train.csv',header = 0);



test_df = pd.read_csv('../data/test.csv',header = 0);



print train_df.info()

print test_df.info()

