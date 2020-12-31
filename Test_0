"""
Table of contents:

1. Pytorch tensors
2. Basic autograd example 1
3. Basic autograd example 2
4. Basic autograd example 3
"""


# Make some imports
import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Pytorch tensors

# Very similar to numpy
x = torch.tensor(1.)
print(x)

# create tensor with values [1,2,3,4,5]
# Float is 32-bit floating-point number
x = torch.tensor([1,2,3,4,5]).float()
print(x)
print(x.size())

# initialize tensor with SHAPE 2x5x5, initialized with random uniform distribution (between (0,1) )
ran = torch.Tensor(2,5,5).uniform_()
print(ran.size())
print(ran)

# min, max, mean, reshaping
ran = torch.Tensor(2,5,5).uniform_()
print(ran)
print(f'size of ran: {ran.size()}') # same as shape in numpy
print(f'min value of ran: {ran.min()}')
print(f'max value of ran: {ran.max()}')
print(f'mean value of ran: {ran.mean()}')

# convert ran shape into (5,2,5) instead of 2,5,5
rsan = ran.view(5,2,5)
"""
2x3 matrix
1,2,3
4,5,6
.view(3,2) =>
1,2
3,4
5,6
.transpose(1,0) (.permute(1,0)) =>
1,4
2,5
3,6
"""
print(f'size/shape of ran: rsan {rsan.size()}')

# Tensor math: uniform random (0,1), multiply by 10 and convert to integer
x = (torch.Tensor(2, 3, 4).uniform_() * 10).int()
y = (torch.Tensor(2, 3, 4).uniform_() * 10).int()
print(x.size())
print(y.size())
print(f'Value of x and y')
print(x)
print(y)
a = x + y
m = x * y
print(f'Value of a and m')
print(a)
print(m)

# transpose of y (the last 2 dimensions), swap dimension 1 and dimension 2
# yt = y.transpose(1, 2)
# print(f'size of yt {yt.size()}') # 2, 4, 3
# print(yt)
# x (2, 3, 4) , yt (2, 4, 3) => matmul (2, 3, 3)
# Note: matmul means matrix multiplication 

yt = y.permute(0,2,1)
print(f'size of yt {yt.size()}')
print(yt)

matmul = torch.matmul(x, yt)
print(f'size of matmul {matmul.size()}')
print(matmul)

# 2. Basic autograd (automatic gradients) example 1

# Create tensors.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# prints None
# print(x.grad)
# print(w.grad)
# print(b.grad)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3: dy/dx = w = 2

# prints None
# print(x.grad)
# print(w.grad)
# print(b.grad)

# Compute gradients.
y.backward() # dy/
# Effect: 
# x.grad += dy/dx
# w.grad += dy/dw
# b.grad += dy/db
# Gradients are accumulated

# Print out the gradients.
print(x.grad)    # x.grad = 2 = dy/dx
print(w.grad)    # w.grad = 1 = dy/dw
print(b.grad)    # b.grad = 1 = dy/db

# variable.grad is accumulated
z = w * x + b
z.backward()
print(x.grad)    # x.grad += 2 ==> becomes 4
print(w.grad)    # w.grad += 1 ==> becomes 2
print(b.grad)    # b.grad += 1 ==> becomes 2

# Setting gradients to 0.
x.grad.data.zero_()
w.grad.data.zero_()
b.grad.data.zero_()

# Print out the gradients.
print(x.grad)    # x.grad ==> becomes 0
print(w.grad)    # w.grad ==> becomes 0
print(b.grad)    # b.grad ==> becomes 0

# Setting and printing gradient again.
y = w * x + b    # y = 2 * x + 3: dy/dx = w = 2
y.backward() # dy/
print(x.grad)    # x.grad = 2 = dy/dx
print(w.grad)    # w.grad = 1 = dy/dw
print(b.grad)    # b.grad = 1 = dy/db

# 3. Basic autograd (automatic gradients) example 2

# Create tensors.
x = torch.tensor(1., requires_grad=True)
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(3., requires_grad=True)
w2 = torch.tensor(4.,  requires_grad=True)
b2 = torch.tensor(1., requires_grad=True)

# Build a computational graph. and chain-rule
y1 = w1 * x + b1    # y1 = 2 * x + 3 = 5   :  dy1/dx = w = 2
y2 = w2 * y1 + b2   # y2 = 4 * 5 + 1 = 21  :  dy2/dx = dy2/dy1 * dy1/dx = 4 * 2 = 8
y3 = y2 ** 2        # y3 = 21^2      = 441 :  dy3/dx = dy3/dy2 * dy2/dy1 * dy1/dx = 2*y2 * dy2/dy1 * dy1/dx = 2*21*8 = 336


print(f'y1 = {y1}')
print(f'y2 = {y2}')
print(f'y3 = {y3}')

# Compute gradients.
y3.backward() # dy/

# Print out the gradients.
print(x.grad)    # x.grad = 2 = dy3/dx
print(w1.grad)    # w1.grad = 1 = dy3/dw1
print(b1.grad)    # b1.grad = 1 = dy3/db1

print(w2.grad)    # w2.grad = 1 = dy3/dw2
print(b2.grad)    # b2.grad = 1 = dy3/db2

# 4. Basic autograd example 3

# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)
print(x)
print(y)

print('='*30)

# Build a fully connected layer. For building a (m x n transformation matrix) of weight variables:
# class nn.Linear():
# def __init__(in_features: int, out_features: int, bias: bool=True) ->None
linear = nn.Linear(3, 2) # y = W*X + b 
print(linear)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Forward pass.
predict = linear(x)
print(f'predict:\n{predict}')

# Build loss function.
# 1/N * sum( (x_i - y_i)^2 )
criterion = nn.MSELoss()

# Compute loss.
loss = criterion(predict, y)
print('loss: ', loss.item())
# print(loss)

# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

print('='*30)

"""Build optimizer:

Now, our next step is to update our parameters. For this purpose, we specify 
the optimizer that uses the gradient descent algorithm. We use SGD() function 
known as stochastic gradient descent for optimization. SGD minimizes the total 
loss one sample at a time and typically reaches convergence much faster as it 
will frequently update the weight of our model within the same sample size.

Here, lr stands for learning rate, which is initially set to 0.01."""

# Option 1: Build optimizer with torch.optim.SGD
# optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
# optimizer.step()      # 1-step gradient descent.

# Option 2: You can also perform gradient descent at the low level.

linear.weight.data.sub_(0.01 * linear.weight.grad.data) # new_weight = old_weight - 0.01 * weight.grad
linear.bias.data.sub_(0.01 * linear.bias.grad.data) # new_weight = old_weight - 0.01 * weight.grad

# Print out the new linear.
print(linear)
print ('New w: ', linear.weight)
print ('New b: ', linear.bias)

# Print out the prediction after 1-step gradient descent.
predict = linear(x)
print(f'New predict:\n{predict}')

# ReLU the output.
relu = nn.ReLU()
relu_predict = relu(predict)
print(f'New relu_predict:\n{relu_predict}')

# Print out the loss after 1-step gradient descent.
loss = criterion(predict, y)
print('loss after 1 step optimization: ', loss.item())

# Set gradients to 0.
# optimizer.zero_grad()     # Either this way
linear.weight.grad.data.zero_()     # Or this way
linear.bias.grad.data.zero_()     # Or this way
# print(linear.weight.grad)
# print(linear.bias.grad)

# Backward pass.
loss.backward()

# Print out the gradients.
print ('New dL/dw: ', linear.weight.grad) 
print ('New dL/db: ', linear.bias.grad)

