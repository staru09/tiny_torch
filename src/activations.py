import numpy as np
from src.tensor import Tensor

class Sigmoid:

    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
       data = x.data
       result = np.zeros_like(data)

       pos_mask = data>=0
       result[pos_mask] = 1/(1+np.exp(-data[pos_mask]))

       neg_mask = data<0
       exp_x = np.exp(data[neg_mask])
       result[neg_mask] = exp_x / (1.0 + exp_x)

       return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    


class ReLU:

    def parameters(self):
        parameters = []

    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0,x.data)
        return Tensor(result)

    def __call__(self, x:Tensor) -> Tensor:
        return self.forward(x)


class Tanh:

    def parameters(self):
        parameters=[]

    def forward(self,x:Tensor)->Tensor:
        x = x.data
        result = np.tanh(x)

        return Tensor(result)

    def __call__(self,x:Tensor)-> Tensor:
        return self.forward(x)
    
class GELU:

    def project(self):
        params=[]

    def forward(self,x:Tensor)->Tensor:

        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        result = sigmoid_part * x.data

        return Tensor(result)

    def __call__(self,x:Tensor)-> Tensor:
        return self.forward(x)
    

class Softmax:
    
    def projections(self):
        return []

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:

        x_max = np.max(x.data,axis=dim,keepdims=True)

        x_shifted = x.data - x_max

        expo_part = np.exp(x_shifted)

        deno_part = np.sum(expo_part,axis=dim,keepdims=True)

        result = expo_part/deno_part

        return Tensor(result)
        
    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        return self.forward(x, dim)