class FCLayer: #fully connected input neurons,output neurons
  def __init__(self,input_size,output_size):
    self.input_size=input_size
    self.output_size=output_size
    self.weights=np.random.randn(input_size,output_size)-0.5
    self.bias=np.random.randn(1,output_size)-0.5
  
  def forward_propagation(self,input_data):
    self.input=input_data
    self.output=np.dot(self.input,self.weights)+self.bias
    return self.output
  
  def backward_propagation(self,output_error,lr):
    input_error=np.dot(output_error,self.weights.T)
    weights_error=np.dot(self.input.T,output_error)
    #bias error is output error

    self.weights-=lr*weights_error
    self.bias-=lr*output_error

    return input_error
  
#define the activation layer  
class ActivationLayer:
  def __init__(self,activation,activation_prime):
    self.activation=activation
    self.activation_prime=activation_prime
  
  def forward_propagation(self,input_data):
    self.input=input_data
    self.output=self.activation(self.input)
    return self.output
  
  def backward_propagation(self,output_error,lr):
    return self.activation_prime(self.input)*output_error
  
#output is softmax layer
class SoftmaxLayer:
  def __init__(self,input_size):
    self.input_size=input_size

  def forward(self,input):
    self.input=input
    temp=np.exp(input)
    self.output=temp/np.sum(temp)
    return self.output
  
  def backward(self,output_error,lr):
    input_error=np.zeros(output_error.shape)
    out=np.tile(self.output.T,self.input_size) #new array by repeating arr as (arr,times)[0 1 2 0 1 2 0 1 2] ([0 1 2] ,3)
    return self.output*np.dot(output_error,np.identity(self.input_size)-out) #identity diagonal 1s all 0s
  
  
  #utility functions
  def tanh(x):
  return np.tanh(x)

def tanh_prime(x):
  return 1-np.tanh(x)**2;

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_prime(x):
  return np.exp(-x)/(1+np.exp(-x))**2

def relu(x):
  return np.maximum(x,0)

def relu_prime(x):
  return np.array(x>=0).astype('int')


#loss function
def mse(y,y_pred):
  return np.mean(np.power(y-y_pred,2))

def mse_prime(y,y_pred):
  return 2*(y_pred-y)/y.size

#solving XOR

x_train=np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
y_train=np.array([[[0]],[[1]],[[1]],[[0]]])

network=[
    FCLayer(2,3),
    ActivationLayer(tanh,tanh_prime),
    FCLayer(3,1),
    ActivationLayer(tanh,tanh_prime)
]

#train

epochs=1000
lr=0.1

for i in range(epochs):
  error=0
  for j in range(len(x_train)):
    output=x_train[j]
    for layer in network:
      output=layer.forward_propagation(output)
    
    error+=mse(y_train[j],output)
    output_error=mse_prime(y_train[j],output)

    for layer in reversed(network):
      output_error=layer.backward_propagation(output_error,lr)
  
  error/=len(x_train)
  print(i+1," ",epochs,"  ",error)


#test prints the epochs

#predict
result=[]
for i in range(len(x_train)):
  output=x_train[i]
  for layer in network:
    output=layer.forward_propagation(output)
  result.append(output)

print(result)
#[array([[0.0276343]]), array([[0.98143119]]), array([[0.98125719]]), array([[0.07065865]])]
