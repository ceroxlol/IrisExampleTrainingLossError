import pandas as pd
import numpy as np

from sklearn import preprocessing, model_selection
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

data = pd.read_csv('Iris.csv')
data = data.drop(['Id'], axis =1)

data = shuffle(data)


i = 8
data_to_predict = data[:i].reset_index(drop = True)
predict_species = data_to_predict.Species
predict_species = np.array(predict_species)
prediction = np.array(data_to_predict.drop(['Species'],axis= 1))

data = data[i:].reset_index(drop = True)

# head value for testing
head = 4

X = data.drop(['Species'], axis = 1).head(head)
X = np.array(X)
Y = data['Species'].head(head)

# Transform name species into numerical values
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)
#print(Y)

# We have 3 classes : the output looks like :
#0,0,1 : Class 1
#0,1,0 : Class 2
#1,0,0 : Class 3

train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.5, random_state = 0)

input_dim = len(data.columns) - 1

model = Sequential()
model.add(Dense(8, input_dim = input_dim , activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(Y.shape[1], activation = 'softmax'))

if Y.shape[1] < 2:
    model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' )
else:
    model.compile(loss='categorical_crossentropy', optimizer='adam')

validation_data = (test_x, test_y)

batch_size = 2

model.fit(train_x, train_y, epochs = 1, batch_size = batch_size, validation_data=validation_data)

print(model.evaluate(train_x, train_y, batch_size))


p = model.predict_proba(train_x, batch_size=batch_size)
ll = log_loss(train_y, p)
print(str(ll))
ll = log_loss(test_y, p)
print(str(ll))
