class SVM():
  def __init__(self,lr=0.01,lambda_param=0.01,n_iters=1000):
    self.lr=lr
    self.lambda_param=lambda_param
    self.n_iters=n_iters
    self.w=None
    self.b=None
  
  def fit(self,X,y):
    n_samples,n_features=X.shape
    y_=np.where(y==0,-1,1)
    self.w=np.zeros(n_features)
    self.b=0

    for i in range(self.n_iters):
      for idx,x_i in enumerate(X):
        condition=y_[idx]*np.dot(x_i,self.w)-self.b >=1
        if condition:
          self.w-=self.lr*(2*self.lambda_param*self.w)
        else:
          self.w-=self.lr*(2*self.lambda_param*self.w-np.dot(x_i,y_[idx]))
          self.b-=self.lr*y_[idx]
    
  def predict(self,X):
    approx=np.dot(X,self.w)-self.b
    return np.sign(approx) #-1 or 1

  
X,y=datasets.make_blobs(n_samples=50,
                        n_features=2,
                        centers=2,
                        cluster_std=1.05,
                        random_state=40)
y=np.where(y==0,-1,1) #if y==0 print -1 else print 1

clf=SVM()
clf.fit(X,y)

predictions=clf.predict(X)
print(clf.w,clf.b)

def get_hyperplane_value(x,w,b,offset):
    return(-w[0]*x+b+offset)/w[1]
def plotsvm():  
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  plt.scatter(X[:,0],X[:,1],marker="o",c=y)

  x01=np.amin(X[:,0])
  x02=np.amax(X[:,0])

  x1_1 = get_hyperplane_value(x01, clf.w, clf.b, 0)
  x1_2 = get_hyperplane_value(x02, clf.w, clf.b, 0)

  x1_1_m = get_hyperplane_value(x01, clf.w, clf.b, -1)
  x1_2_m = get_hyperplane_value(x02, clf.w, clf.b, -1)

  x1_1_p = get_hyperplane_value(x01, clf.w, clf.b, 1)
  x1_2_p = get_hyperplane_value(x02, clf.w, clf.b, 1)

  ax.plot([x01, x02], [x1_1, x1_2], "y--")
  ax.plot([x01, x02], [x1_1_m, x1_2_m], "k")
  ax.plot([x01, x02], [x1_1_p, x1_2_p], "k")

  x1_min = np.amin(X[:, 1])
  x1_max = np.amax(X[:, 1])
  ax.set_ylim([x1_min - 3, x1_max + 3])
  
  plotsvm()
