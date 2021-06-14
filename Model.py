import pandas as pd
import matplotlib.pyplot as plt
import pickle 


train = pd.read_csv("train.csv",index_col=0)
train['Failure'] = train['Failure'].astype('category')
train['Failure'] = train['Failure'].cat.codes

#train['Humidity'][train['Humidity']< 70] = 70
#train['Humidity'][train['Humidity']>95] = 95

#train['Temperature'][train['Temperature']<60] = 60
#train['Temperature'][train['Temperature']>71.875] = 71.875

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = train[['Temperature', 'Humidity' ]] #'Hours_since_prev', 'Year', 'Month', 'Day', 'Week'
y = train['Failure']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=101)

model = LogisticRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)
#saving model to disk
with open('model.pkl','wb') as f:
   pickle.dump(model,f)
#pickle.dumb(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[60, 70]]))