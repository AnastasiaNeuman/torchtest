import torch
import numpy as np

#tensors are used to encode the inputs, outputs, and parameters of a model in pytorch

#initializing a tensor

#directly from data
#the data type is automatically inferred
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#from a numpy array (numpy arrays can also be initialized from tensors)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#from another tensor
#he new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#with random or constant values
#shape is a tuple of tensor dimensions.

shape = (2,3,) #determines dimensionality of the tensor
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#printing tensor attributes
tensor = torch.rand(3,4)
print(tensor)
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#by default, tensors are created on cpu
#have to explicitly move tensors to the accelerator using .to method

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

#tensor operations very similar to numpy operations

#ex: indexing and slicing 
tensor = torch.ones(4, 4)
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor) 

#joining tensors, can also use torch.stack
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#arithmetic
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item():

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
#acts as an overwriting function
#the use is discourages - saves memory but problematic for derivatives
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)






