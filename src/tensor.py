import numpy as np

class Tensor:
    def __init__(self,data):
        self.data = np.array(data,dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        """Human-readable string representation."""
        return f"Tensor({self.data})"

    def __add__(self,other):
        if isinstance(other,Tensor):
            return Tensor(self.data+other.data)
        else:
            return Tensor(self.data+other)

    def __sub__(self,other):
        if isinstance(other,Tensor):
            return Tensor(self.data-other.data)
        else:
            return Tensor(self.data-other)

    def __mul__(self,other):
        if isinstance(other,Tensor):
            return Tensor(self.data*other.data)
        else:
            return Tensor(self.data*other)

    def __truediv__(self,other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)
    
    def __getitem__(self,key):
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
                result_data = np.array(result_data)
        return Tensor(result_data)


    def matmul(self,other):
        if not isinstance(other,Tensor):
            raise TypeError("Can't perform calculation on this")
            
        if len(self.shape) == 0 or len(other.shape) == 0:
            raise ValueError("Need atleast 1-D vector for matmul")

        if len(self.shape) >= 2 and len(other.shape)>= 2:

            if self.shape[-1]!=other.shape[-2]:
                raise ValueError("Dimension mismatched for matmul")

        if len(self.shape) == 2 and len(other.shape) == 2:
            M, K = self.shape
            K2, N = other.shape
            result = np.zeros((M, N), dtype=np.float32)
            
            for i in range(M):
                for j in range(N):
                    result[i, j] = np.dot(self.data[i, :], other.data[:, j])
            
            return Tensor(result)

        else:
            return Tensor(np.matmul(self.data,other.data))

    def __matmul__(self,other):
        return self.matmul(other)


    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape 

        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")
        
            new_shape = list(new_shape)
            idx = new_shape.index(-1)
            
            known_product = 1
            for i, dim in enumerate(new_shape):
                if i != idx: 
                    known_product *= dim
            
            unknown_dim = self.size // known_product
            new_shape[idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape)!=self.size:
            raise ValueError(f"Total elements must match, {self.size}!={int(np.prod(new_shape))}")
        result = np.reshape(self.data, new_shape)

        return Tensor(result)


    def transpose(self, dim0=None, dim1=None):
        # Case 1: No dimensions specified - swap last two
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        # Case 2: Only one dimension specified - error
        elif dim0 is None or dim1 is None:
            raise ValueError("Both dim0 and dim1 must be specified")
        # Case 3: Both dimensions specified
        else:
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        
        return Tensor(transposed_data)

    def sum(self,axis=None, keepdims=False):
        result = np.sum(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)

    def mean(self,axis=None, keepdims=False):
        result = np.mean(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)

    def max(self,axis=None, keepdims=False):
        result = np.max(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)