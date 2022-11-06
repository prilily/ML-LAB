from sklearn.datasets import make_classification

X,y =make_classification(n_features=2,n_redundant=0,n_informative=2,random_state=1,n_clusters_per_class=1)

plt.figure(figsize=(10,8))
plt.plot(X[:,0][y==0],X[:,1][y==0],"g^")
plt.plot(X[:,0][y==1],X[:,1][y==1],"bs")

def normalize(X):
  m,n=X.shape
  for i in range(n):
    X=(X-X.mean(axis=0))/X.std(axis=0)

def loss(y,y_hat):
  loss=-np.mean(y*(np.log(y_hat))-(1-y)*np.log(1-y))
  return loss

def sigmoid(z):
  return 1.0/(1+np.exp(-z))

def train(X,y,epochs,lr):
  #bs- batch size
  m,n=X.shape
  w=np.zeros((n,1))
  b=0
  y=y.reshape(m,1)
  x=normalize(X)

  losses=[]

  for epoch in range(epochs):
    y_hat=sigmoid(np.dot(X,w)+b)
    dw=(1/m)*np.dot(X.T,(y_hat-y))
    db=(1/m)*np.sum((y_hat-y))

    w-=lr*dw
    b-=lr*db

    l=loss(y,y_hat)
    losses.append(l)
  return w,b,losses



def predict(X,w,b):
  x=normalize(X)
  pred=sigmoid(np.dot(X,w)+b)
  pred_class=[]

  pred_class=[1 if i>0.5 else 0 for i in pred]
  return np.array(pred_class)

def accuracy(y,y_hat):
  acc= np.sum(y==y_hat)/len(y)
  return acc

def plot_boundary(X,w,b):
  #equate mx+c=wX+b so find m and c
  x1=[min(X[:,0]),max(X[:,0])]
  m=-w[0]/w[1]
  c=-b/w[1]
  x2=m*x1+c

  fig=plt.figure(figsize=(10,8))
  plt.plot(X[:,0][y==0],X[:,1][y==0],"g^")
  plt.plot(X[:,0][y==1],X[:,1][y==1],"bs")
  plt.xlim([-2,2])
  plt.ylim([0,2.2])
  plt.plot(x1,x2,'y-')
  

w,b,l=train(X,y,epochs=300,lr=0.01)
plot_boundary(X,w,b)

x_pred=predict(X,w,b)
accuracy(X,x_pred)
