class SVM():
  def __init__(self,lr=0.001,lambda_param=0.01,n_iters=1000):
    self.lr=lr
    self.lambda_param=lambda_param
    self.n_iters=n_iters
    self.w=None
    self.b=None

  def fit(self,X,y):
    n_samples,n_features=X.shape
    self.w=np.zeros(n_features)
    self.b=0

    y_=np.where(y<=0,1,-1)

    for _ in range(self.n_iters):
      for idx, x_i in enumerate(X):
          condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
          if condition:
              self.w -= self.lr * (2 * self.lambda_param * self.w)
          else:
              self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])  )
              self.b -= self.lr * y_[idx]

  
  def predict(self,X):
    return np.sign(np.dot(X,self.w)-self.b)
  
X,y=datasets.make_blobs(n_samples=50,n_features=2,centers=2,cluster_std=1.05,random_state=40)
y=np.where(y==0,1,-1)
X.shape

clf = SVM()
clf.fit(X, y)

pred=clf.predict(X)

#to plot
def visualise():
  def hyperplane(x,w,b,offset):
    return (-w[0]*x+b+offset)/w[1]

  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  plt.scatter(X[:,0],X[:,1],marker="o",c=y)

  #get the margin values
  x01=np.amin(X[:,0])
  x02=np.amax(X[:,0])

  #central line
  x11=hyperplane(x01,clf.w,clf.b,0)
  x12=hyperplane(x02,clf.w,clf.b,0)

  #for -ve class
  xm1=hyperplane(x01,clf.w,clf.b,-1)
  xm2=hyperplane(x02,clf.w,clf.b,-1)

  #for +ve class
  xp1=hyperplane(x01,clf.w,clf.b,1)
  xp2=hyperplane(x02,clf.w,clf.b,1)

  ax.plot([x01,x02],[x11,x12],"y--")
  ax.plot([x01,x02],[xm1,xm2],"k")
  ax.plot([x01,x02],[xp1,xp2],"k")

  x1_min=np.amin(X[:,1])
  x1_max=np.amax(X[:,1])

  ax.set_ylim([x1_min-3,x1_max+3])

  plt.show()

visualise()
