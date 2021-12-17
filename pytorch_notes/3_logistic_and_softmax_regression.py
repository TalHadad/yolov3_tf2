######################################
# 1. Logistic Regression Prediction
######################################
import torch.nn as nn

# 1.1.
sig = nn.Sigmoid()
y_hat = sig(z)

# 1.2.
import torch.nn.functional as F # or from torch import sigmoid (?)
y_hat = F.sigmoid(z) # actual function

# 1.3.
model = nn.Sequential(nn.Linear(1,1), nn.Sigmoid()) # sigmoid(xw+b)
y_hat = model(x)

# 1.4. Custom Module
import torch.nn as nn
class logistic_regression(nn.Module):
      def __init__(self, in_size, out_size):
          super(logistic_regression, self).__init__()
          self.linear = nn.Linear(in_size, out_size)
      def forward(self,x):
          return F.sigmoid(self.linear(x))

######################################
# 2. Training Logistic Regression
######################################
# Logistic Regression: P(Y|tetha,x) = mul{n=1,N}((sigmoid(w*x_n+b)^y_n)*(1-sigmoid(w*x_n+b)^(1-y_n)))
# max(p) = max(log(p)) = min(-log(p)) => min(-(1/N)log(p))
# Cross Entropy: l(w) = -(1/N)sum{n=1,N}(y_n*ln(sigmoid(w*x_n+b))+(1+y_n)*ln(1-sigmoid(w*x_n+b)))

# 2.1.
def loss_function(y_hat,y):

######################################
# 3. Softmax Regression
######################################
