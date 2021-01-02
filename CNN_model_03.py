# 1. Download the dataset

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784  # the image has 784 pixel
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='.', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='.', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

2. Visualising the image

image, label = test_dataset[10]
# reduce batch=1 to no batch
image = image[0]
print(f'{image.size()} , Label: {label}')
plt.imshow(image)

# 3. Initiate the Neural Network (multi-layer perceptron)
# The network has 2 layers, with ReLu activation in between

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

mnist_size = (1, 28, 28)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out) # Dropout
        out = self.fc2(out)
        out = self.bn(out) # BatchNorm
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Convolution Neural Network

"""
Note: if a layer has dimension [100, 1, 28, 28]:

- The first dimension, 100, denotes the number of photos in consideration, 
  meaning there are 100 photos under our analysis;

- The second dimension, 3, denotes the number of layers or filters / channels 
  of each pixel, 
  - e.g. if a coloured photo has values for R, G, B colour, it has 3 layers, R, G, B, 
    where the number in each layer denotes the value of the respective colour, 
  - however in our specific case, the initial photo has only 1 single layer, 
    i.e. it does not have R, G, B feature, only labelling of feature, e.g. purple, 
    dark blue, light blue, dark green, light green, yellow, therefore, in terms 
    of any incoming photos, this dimension should only have a value of 1,
  - e.g. if a layer has 16 filters / channels, then the second dimension will be 16;

- The third and fourth dimension, 28 and 28, denote the number of pixels in the 
  height and width of the photo, or the number of activations in the height and 
  width of the convolution layer, 
  - e.g. if a photo has dimension (28 x 28), it has 28 pixels in its height and 
    28 pixels in its width, so the photo has 28 x 28 = 784 pixels in total,
  - e.g. if a convolution layer has dimension (12 x 12), it has 12 activations 
    in its height and 12 activations in its width, so the convolution layer has 
    12 x 12 = 224 activations in total.
"""

class CNN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
    """kernel_size is [1, 3, 3] (note that the initial photo has only 1 single 
    R, G, B layer), so the regressor of the kernel has 9 input parameters, 
    and 9 weights associated with those 9 input parameters. Together with a bias 
    input (an input parameter of 1 in the regressor) and a bias weight, there 
    are 10 weights in total.
    Without zero padding, the initial (28 x 28) pixel photo will reduce its 
    width and height dimension to become a (26 x 26) convolution layer.
    Equation: width = (width - kernel_size) / stride + 1 = (28 - 3) / 1 + 1 = 26
    height = (height - kernel_size) / stride + 1 = (28 - 3) / 1 + 1 = 26
    Because out_channels = 8, this first layer of convolution will stochastically 
    (probabilistically) produce 8 layer of filters / channels.
    So the activation produced by this convolutional neural network is a 
    [number_of_photos, 8, 26, 26] dimensional tensor."""

    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2), stride=1)
    """kernel_size is [8, 2, 2] (note that from the first convolutional neural 
    network, the input activation has 8 channels), so the regressor of the kernel 
    has 8 x 2 x 2 = 32 input parameters, and 32 weights associated with those 
    32 input parameters. Together with a bias input (an input parameter of 1 in 
    the regressor) and a bias weight, there are 33 weights in total.
    Without zero padding, the (26 x 26) input layer will reduce its 
    width and height dimension to become a (25 x 25) convolution layer.
    Equation: width = (width - kernel_size) / stride + 1 = (26 - 2) / 1 + 1 = 25
    height = (height - kernel_size) / stride + 1 = (26 - 2) / 1 + 1 = 25
    Because out_channels = 16, this first layer of convolution will stochastically 
    (probabilistically) produce 16 layer of filters / channels.
    So the activation produced by this convolutional neural network is a 
    [number_of_photos, 16, 25, 25] dimensional tensor."""

    self.pool1 = nn.MaxPool2d(kernel_size=2)
    """kernel_size is [16, 2, 2] (note that from the second convolutional neural 
    network, the input activation has 16 channels).
    With max pooling, the (25 x 25) input layer will reduce its 
    width and height dimension to become a (12 x 212) convolution layer.
    Equation: width = int(width/2) = int(25/2) = int(12.5) = 12
    height = int(height/2) = int(25/2) = int(12.5) = 12
    So the activation produced by this max pooling layer is a 
    [number_of_photos, 16, 12, 12] dimensional tensor."""

    # self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(2, 2), stride=1)
    """Why need a convolution that reduces depth / reduces number of filters?"""

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(8)

    # Linear regression at the last step.
    # self.out_linear = nn.Linear(self._derive_last_dim(input_size), num_classes)   # We need to calculate the input_size.
    # Alternatively:
    self.out_linear = nn.Linear(self._derive_last_dim(2304), num_classes)   # Easier.
  
  def _derive_last_dim(self, input_size):   # Calculating the input_size for the last step of linear regression.
    # size = [1] + list(input_size)
    # print(size)                           # Will return an error?
    # y = torch.Tensor(*size).uniform_()    # Create a (1 x 1 x 28 x 28 tensor of which all elements are uniformly distributed).
    # print(f'in_y = {size} {y.size()}')    # Observe the dimension of y: in_y = [1, 1, 28, 28] torch.Size([1, 1, 28, 28]).

    y = torch.Tensor(1,1,28,28).uniform_()  # Wouldn't it be easier?
    print(f'y.size() = {y.size()}')         # Observe the dimension of y: y.size() = torch.Size([1, 1, 28, 28]).
    """The first dimension, 1, is unimportant, as we only want to know how many 
    weight parameters we need in the last step of linear regression, regardless 
    of how many input photos we are analyzing.
    The second, third and fourth dimension, [1, 28, 28], refer to the fact that 
    we are analysing photos of (28 x 28) pixels with only 1 single R, G, B layer."""

    y = self.conv_forward(y)                # Test out convoluting y.
    print(f'y.size() = {y.size()}')         # Observe the dimension of y: y.size() = torch.Size([1, 16, 12, 12]).
    """The first dimension, 1, is unimportant, as we only want to know how many 
    weight parameters we need in the last step of linear regression, regardless 
    of how many input photos we are analyzing.
    The second, third and fourth dimension, [16, 12, 12], refer to the fact that 
    the activation produced by the convolutional neural network is a 
    [16, 12, 12] dimensional tensor."""

    y = y.view(y.size(0), -1)               # Make y becomes a (1 x 2304 array), 16 x 12 x 12 = 2304.
    """Lining up the second, third and fourth dimension together, 
    noting that 16 x 12 x 12 = 2304. So in the last step of linear regression, 
    there are 2304 input parameters, and 2304 weights associated with those 
    2304 input parameters. Together with a bias input (an input parameter of 1 in 
    the regressor) and a bias weight, there are 2305 weights in total."""

    return y.size(1)                        # Return the column size, 2304.
    """2304 is the number of weights, which is the 'input_size', needed in the 
    last step of the linear regression."""

  def conv_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.dropout(x)

    # x = self.conv3(x)
    # x = self.bn2(x)
    # x = self.relu(x)

    x = self.pool1(x)
    return x
  
  def forward(self, x):
    x = self.conv_forward(x)
    # print(f'x.size() = {x.size()}')                     # This step is essential as we can verify the dimension of the tensor x.
    """Very useful for verifying the number of weights / 'input_size' needed 
    in the last step of the linear regression.
    Note the followings:
    1. x.size() can only be printed in the next phase of 'Train the network',
    2. x.size() will be printed numerous times. This is because the 'train_loader' 
    has 600 data, the 'test_loader' has 100 data, and we are iterating for 20 epochs."""
    x = x.view(x.size(0), -1)           # x becomes a (100 x 2304 matrix) 
    """The first dimension, 100, refers to the number of photos in analysis. 
    As there are 100 number of images in each 'images' pack of 'train_loader'
    and in each 'images' pack of 'test_loader' data, our neural network is 
    analysing 100 photos simultaneously, therefore the first dimension is 100.
    The second, third and fourth dimension are lined up together, and noting that 
    16 x 12 x 12 = 2304, there are 2304 is the number of weights, which is the 
    'input_size', needed in the last step of the linear regression."""
    x = self.out_linear(x)
    return x

# x = torch.Tensor((1,1,28,28))
# print(x.size())
# print(x)

model = CNN(mnist_size, num_classes).to(device)
print(model)

# 4. Train the network

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# Create SGD (stochastic gradient descent) optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
learning_rate = 0.0001
num_epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Using Adam optimizer.

def evaluate_test(model):
  with torch.no_grad():
    model.eval()
    test_correct = 0
    test_total = 0
    losss = []
    for images, labels in test_loader:
      """test_loader has 100 data. In each of the 100 data, the 'images' pack 
      contains 100 images, while the 'labels' pack also contains 100 labels. 
      Total there are 100 X 100 = 10,000 images and 10,000 respective labels"""
      images = images.reshape(-1, 1, 28, 28).to(device)   # train_loader on CPU, so send it to GPU
      labels = labels.to(device)   # train_loader on CPU, so send it to GPU
      outputs = model(images)
      max_scores, predicted = torch.max(outputs.data, 1)
      loss = criterion(outputs, labels)     # (1 x 1 scalar)
      """The loss function is a Cross Entropy function. It compares the 
      (100 x 10 'outputs' matrix) against the (100 x 1 'labels' vector).
      'loss' is a (1 x 1 scalar)."""

      test_total += labels.size(0)
      test_correct += (predicted == labels).sum().item()
      losss.append(loss.item())   # loss[] will append to become a (1 x 100 array).
    test_loss = sum(losss) / len(losss)  # Turns loss[] into a (1 x 1 scalar).
    test_accuracy = (test_correct / test_total) * 100
  return test_loss, test_accuracy
  # print('Accuracy of the network on the 10000 test images: {} %'.format(test_correct / test_total))

# Train the model
total_step = len(train_loader)    # Should return 600 because train_loader has 600 data.

# Start Training
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(num_epochs):
  # training
  model.train()
  train_correct = 0
  train_total = 0
  train_loss = []
  for i, (images, labels) in enumerate(train_loader):    # train_loader has 600 data.
    """train_loader has 600 data. In each of the 600 data, the 'images' pack 
    contains 100 images, while the 'labels' pack also contains 100 labels. 
    Total there are 600 x 100 = 60,000 images and 60,000 respective labels"""
    """enumerate returns the index and the data, 
    so the 'i' in i, (images, labels) is the index"""
    # Move tensors to the configured device.
    images = images.reshape(images.size(0), 1, 28, 28).to(device)   # train_loader on CPU, so send it to GPU
    """return each image as 1-D array, each image becomes a (1 x 756 
    horizontal vector). As there are 100 images, 'images' becomes a 
    (100 x 756 matrix)."""
    labels = labels.to(device)    # train_loader on CPU, so send it to GPU
    """Each label is a (1 x 1 scalar). As there are 100 labels, 'labels' 
    becomes a (100 x 1 vector)."""
    
    # Forward pass.
    outputs = model(images)     # outputs is a (100 x 10 matrix)  
    max_scores, predicted = torch.max(outputs.data, 1)
    train_total += labels.size(0)
    train_correct += (predicted == labels).sum().item()

    # Computing Cross Entropy Loss.
    loss = criterion(outputs, labels)
    """The loss function is a Cross Entropy function. It compares the 
    (100 x 10 'outputs' matrix) against the (100 x 1 'labels' vector).
    'loss' is a (1 x 1 scalar)."""
    
    # Backward and optimize:

    # zero_grad will zero out all the previous gradient store in the parameters, w/o this, the gradients will be added to the old gradients
    optimizer.zero_grad()
    # perform backward backpropagation
    loss.backward()
    # update the parameters withg gradient descent
    optimizer.step()

    # Update train_loss list.
    train_loss.append(loss.item())
    """Note: In our previous model, we append the train_loss list for every 
    100 data in train_loader (remember train_loader has 600 data). Now in this 
    model, we append the train_loss list for every single data out of the 600 
    train_loader data."""
  
  # Turns the (1 x 600 array) into a single (1 x 1 scalar) of its average value.
  train_loss = sum(train_loss) / len(train_loss)
  train_accuracy = (train_correct / train_total) * 100
  # Testing using the evaluate_test function.
  test_loss, test_accuracy = evaluate_test(model)

  print('Epoch [{}/{}], Train loss: {:.4f}, Test loss: {:.4f}, Train accuracy: {:.4f}%, Test accuracy: {:.4f}%'.format(epoch+1, num_epochs, train_loss, test_loss, train_accuracy, test_accuracy))
  train_losses.append(train_loss)
  test_losses.append(test_loss)
  train_accuracies.append(train_accuracy)
  test_accuracies.append(test_accuracy)

# Has moved to the next section:
"""
# Plotting curve:
print('-' * 30)
# x_axis = list(range(len(train_losses)))
x_axis = list(range(1,num_epochs+1))

# Plot the loss curve
print(f'Loss Curve: Red = train loss, Blue = test loss')
plt.plot(x_axis, train_losses, '-r', x_axis, test_losses, 'b')

# Plot the accuracy curve
print(f'Loss Curve: Red = train accuracy, Blue = test accuracy')
plt.plot(x_axis, train_accuracies, '-r', x_axis, test_accuracies, 'b')
"""

# Compute loss and accuracy
test_loss, test_accuracy = evaluate_test(model)
print('Final loss: {:.4f}, Final accuracy: {:.4f}%'.format(test_loss, test_accuracy))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.pt')

# 5. Plotting the Loss Curve

print(f'Loss Curve: Red = train loss, Blue = test loss')
x_axis = list(range(1,num_epochs+1))
plt.plot(x_axis, train_losses, '-r', x_axis, test_losses, 'b')

# 6. Plotting the Accuracy Curve

print(f'Loss Curve: Red = train accuracy, Blue = test accuracy')
x_axis = list(range(1,num_epochs+1))
plt.plot(x_axis, train_accuracies, '-r', x_axis, test_accuracies, 'b')

# 7. Test the network

# reduce batch=1 to no batch
image = image[0].view(1, 1, 28, 28)
print(f'{image.size()} , Label: {label}')

with torch.no_grad():
  model.eval()
  image = image.to(device)   # train_loader on CPU, so send it to GPU
  layer1 = model.layers[0](image)
  layer2 = model.layers[1](layer1)
  layer3 = model.layers[2](layer2)

plt.figure()
plt.imshow(image.view(28, 28).cpu())

plt.figure()
# plt.imshow(layer1[0].mean(0).permute(1, 2, 0).cpu())
plt.imshow(layer1[0].mean(0).cpu())
plt.figure()
plt.imshow(layer2[0].mean(0).cpu())
plt.figure()
plt.imshow(layer3[0].mean(0).cpu())
