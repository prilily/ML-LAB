import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn import datasets

data={'Hours':[2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8],'Scores':[21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]}
df=pd.DataFrame(data)
df

# x-feature and y is target

#plot
X=np.array(df['Hours']).reshape(-1,1)#column
y=np.array(df['Scores'])
plt.figure(figsize=(8,6))
plt.scatter(X,y)
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

#hypothesis h(x)=wX+b y_hat=np.dot(X,weights(slope))+bias(y-intercept)
# loss function: loss = np.mean((y_hat - y)**2)
X.shape,y.shape

n=1
m=25
weights=np.zeros((n,1))
bias=0

class LinReg:
  def __init__(self,lr=0.01,epochs=500):
    self.lr=lr
    self.epochs=epochs
    self.weights=None
    self.bias=None
  
  def fit(self,X,y):
    m,n=X.shape
    self.weights=np.zeros((n,1))
    self.bias=0

    y=y.reshape(m,1)
    losses=[]

    #gradient descent
    for epoch in range(self.epochs):
      y_hat=np.dot(X,self.weights)+self.bias
      loss=np.mean((y_hat-y)**2)
      losses.append(loss)

      #derivatives of parameters
      dw=(1/m)*np.dot(X.T,(y_hat-y))
      db=(1/m)*np.sum((y_hat-y))

      #updating parameters
      self.weights-=self.lr*dw
      self.bias-=self.lr*db
    
    return self.weights,self.bias,losses
  

  def predict(self,X):
    return np.dot(X,self.weights)+self.bias
  
  x_train,x_test,y_train,y_test=X[:20],X[20:],y[:20],y[20:]
  
model=LinReg(epochs=500)
w,b,l=model.fit(x_train,y_train)

#plot predictions
fig=plt.figure(figsize=(8,6))
plt.scatter(X,y)
plt.plot(X,model.predict(X))
plt.show()

x_pred=model.predict(x_test)
x_pred.flatten()

res=pd.DataFrame({'Actual':y_test,'Predicted':x_pred.flatten()})
res


# Actual	Predicted
# 0	30	28.440968
# 1	54	48.607025
# 2	35	39.004140
# 3	76	68.773081
# 4	86	77.415677


r2_score(x_pred,y_test) 
#0.895610128931885


mean_squared_error(x_pred,y_test)
#34.69337270618255
