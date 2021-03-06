import pandas as pd
import numpy as np
import pylab as P

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../data/train.csv', header=0)


df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']



for i in range(0,2):
	for j in range(0,3):
		df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1) ,'AgeFill'] = median_ages[i,j];


df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass



df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
df = df.drop(['Age'], axis=1)



train_data = df.values


