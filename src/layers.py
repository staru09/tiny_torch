from src.tensor import Tensor
from src.activations import *
import numpy as np


XAVIER_SCALE_FACTOR = 1.0
DROPOUT_MIN_PROB = 0.0
DROPOUT_MAX_PROB = 1.0

class Layer:
    
    def forward(self, x):

        raise NotImplementedError(
            f"forward() not implemented in {self.__class__.__name__}\n"
        )
    
    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        
        return []

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Linear(Layer):
    def __init__(self,in_features,out_features,bias=True):
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(XAVIER_SCALE_FACTOR / in_features)

        weight_data = np.random.randn(in_features,out_features)*scale

        self.weight = Tensor(weight_data)

        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None

    def forward(self,x):
       
        y = x.matmul(self.weight)

        if self.bias is not None:
            y = y+self.bias

        return y


    def parameters(self):
        parameters = [self.weight]
        if self.bias is not None:
            parameters.append(self.bias)

        return parameters

    def __repr__(self):
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"
    

class Dropout(Layer):
    
    def __init__(self,p=0.05):
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(
                f"Invalid dropout probability: {p}\n"
            )
        else:
            self.p = p

    def forward(self,x,training=True):
        if not training or self.p == DROPOUT_MIN_PROB:
            return x
        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data))
        keep_prob = 1.0-self.p

        mask = np.random.random(x.data.shape) < keep_prob

        mask_tensor = Tensor(mask.astype(np.float32))

        scaled = Tensor(np.array(1 / keep_prob))

        output = x*scaled*mask_tensor

        return output

    def __call__(self,x,training=True):
        return self.forward(x,training)

    def __parameters__(self):
        return []
    
    def __repr__(self):
         return f"Dropout(p={self.p})"
    
class Sequential:
    def __init__(self,*layers):
        if len(layers)==1 and isinstance(layers[0],(list,tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self,x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"