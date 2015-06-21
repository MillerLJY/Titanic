import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

#load the train data
train_df = pd.read_csv('../data/train.csv',header = 0);

#transform sex to Gender
train_df['Gender'] = train_df['Sex'].map({'female':0,'male':1}).astype(int);

if(len(train_df.Embarked.isnull()) > 0 ):
	train_df.Embarked[train_df['Embarked'].isnull()] = train_df.Embarked.dropna().mode().values 

Ports = list(enumerate(np.unique(train_df['Embarked'])))
Ports_dict = { name:i for i,name in Ports}
train_df.Embarked = train_df.Embarked.map(lambda x: Ports_dict[x]).astype(int);


# transform Fare
step_fare = 10;
max_fare = 59;
train_df['Fare_level'] = 0;
train_df.Fare[train_df['Fare'] > max_fare] = max_fare; 
for i in range(0,7):
	train_df.Fare_level[(train_df['Fare'] >= i*step_fare) & (train_df['Fare'] < (i+1)*step_fare)] = i;


#fill Age
median_age = np.zeros((2,3))
for i in range(0,2):
	for j in range(0,3):
		median_age[i,j] = train_df.Age[(train_df['Gender'] == i) & (train_df['Pclass'] == j+1)].dropna().median();

for i in range(0,2):
	for j in range(0,3):
		train_df.loc[(train_df['Age'].isnull()) & (train_df['Gender'] == i) & (train_df['Pclass'] == j+1) , 'Age'] = median_age[i,j];
		

step_age = 10;
max_age = 59;
train_df['Age_level'] = 0;
train_df.Age[train_df['Age_level'] > max_age] = max_age;
for i in range(0,7):
        train_df.Age_level[(train_df['Age'] >= i*step_age) & (train_df['Age'] < (i+1)*step_age)] = i;

train_df = train_df.drop(['Name','Sex','Ticket','Cabin','PassengerId','Fare','Age'] , axis=1)


test_df = pd.read_csv('../data/test.csv',header = 0);
#transform sex to Gender
test_df['Gender'] = test_df['Sex'].map({'female':0,'male':1}).astype(int);
test_df.Embarked = test_df.Embarked.map(lambda x: Ports_dict[x]).astype(int);


#fill fare in test
median_fare = np.zeros((2,3))
for i in range(0,2):
	for j in range(0,3):
		median_fare[i,j] = test_df.Fare[ (test_df['Gender'] == i) & (test_df['Pclass'] == j+1)].dropna().median();


for i in range(0,2):
	for j in range(0,3):
		test_df.loc[ (test_df['Fare'].isnull()) & (test_df['Gender'] == i) & (test_df['Pclass']== j+1) , 'Fare' ] = median_fare[i,j];


#transform fare in test
test_df['Fare_level'] = 0;
test_df.Fare[test_df['Fare'] > max_fare] = max_fare;

for i in range(0,5):
        test_df.Fare_level[(test_df['Fare'] >= i*step_fare) & (test_df['Fare'] < (i+1)*step_fare)] = i;



#fill  age in test
for i in range(0,2):
        for j in range(0,3):
                median_age[i,j] = test_df.Age[ (test_df['Gender'] == i) & (test_df['Pclass'] == j+1)].dropna().median();

for i in range(0,2):
        for j in range(0,3):
                test_df.loc[(test_df['Gender'] == i) & (test_df['Pclass'] == j+1) , 'Age'] = median_age[i,j];


test_df['Age_level'] = 0;
test_df.Age[test_df['Age_level'] > max_age] = max_age;
for i in range(0,7):
        test_df.Age_level[(test_df['Age'] >= i*step_age) & (test_df['Age'] < (i+1)*step_age)] = i;

ids = test_df['PassengerId'].values
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Fare','Age'], axis=1)


# the data is now ready to go , So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

print 'Training ...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])


print 'Predicting ...'
output = forest.predict(test_data).astype(int)


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'



#data overview
print train_df.info()
print test_df.info()
