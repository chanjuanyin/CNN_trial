# Train a simple MNIST Neural nets

# 1. Download the dataset

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784  # the image has 784 pixel = 28*28 image (hand-written digit)
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

print(len(train_loader))    # train_loader has 600 data
"""train_loader has 600 data. In each of the 600 data, the 'images' pack 
contains 100 images, while the 'labels' pack also contains 100 labels. Total 
there are 600 X 100 = 60,000 images and 60,000 respective labels"""

for i, (images, labels) in enumerate(train_loader):
        """enumerate returns the index and the data, 
        so the 'i' in i, (images, labels) is the index"""
        # Move tensors to the configured device
        print(len(images))    # In each of the 600 data, the 'images' pack contains 100 images
        print(len(labels))    # In each of the 600 data, the 'labels' pack contains 100 labels
        if i==0:
          break

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

print(len(test_loader))    # test_loader has 100 data
"""test_loader has 100 data. In each of the 100 data, the 'images' pack 
contains 100 images, while the 'labels' pack also contains 100 labels. Total 
there are 100 X 100 = 10,000 images and 10,000 respective labels"""

# 2. Visualising the image

# Visualising the image.
image, label = test_dataset[10]
# reduce batch=1 to no batch
image = image[0]
print(f'{image.size()} , Label: {label}')
plt.imshow(image)

# 3. Initiate the Neural Network (multi-layer perceptron)
# The network has 2 layers, with ReLu activation in between

"""Device configuration
cuda means GPU, if torch.cuda.is_available() then we run on cuda which is GPU
otherwise we run on CPU"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001 # 0.01 0.001 0.0001 also can do


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        """(100 x 756 'image' matrix regressor) X (756 coordinates x 500 columns)
        = (100 x 500 matrix)"""
        out = self.relu(out)  # ReLU the output, which is the (100 x 500 matrix)
        out = self.fc2(out)
        """(100 x 500 ReLU-ed matrix regressor) X (500 coordinates x 10 columns)
        = (100 x 10 matrix)"""
        return out     # (100 x 10 matrix)

model = NeuralNet(input_size = input_size, hidden_size = hidden_size, num_classes = num_classes).to(device)
print(model)

# 4. Train the network

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# Cross entropy loss criterion will:
#   1. take the output (fc2) and make softmax of the score -> probability p(x)
#   2. compute cross-entropy loss with Loss = sum( y_i * log p(x) )


# Create SGD (stochastic gradient descent) optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)

losses = []

# for images, labels in train_loader
# for index, (images, labels) in enumerate(train_loader):


for epoch in range(num_epochs):       # num_epochs = 5
    for i, (images, labels) in enumerate(train_loader):  # train_loader has 600 data
        """train_loader has 600 data. In each of the 600 data, the 'images' pack 
        contains 100 images, while the 'labels' pack also contains 100 labels. 
        Total there are 600 x 100 = 60,000 images and 60,000 respective labels"""
        """enumerate returns the index and the data, 
        so the 'i' in i, (images, labels) is the index"""
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)   # train_loader on CPU, so send it to GPU
        """return each image as 1-D array, each image becomes a (1 x 756 
        horizontal vector). As there are 100 images, 'images' becomes a 
        (100 x 756 matrix)."""
        labels = labels.to(device)    # train_loader on CPU, so send it to GPU
        """Each label is a (1 x 1 scalar). As there are 100 labels, 'labels' 
        becomes a (100 x 1 vector)."""
        
        # Forward pass
        outputs = model(images)     # outputs is a (100 x 10 matrix)
        loss = criterion(outputs, labels)   # (1 x 1 scalar)
        """The loss function is a Cross Entropy function. It compares the 
        (100 x 10 'outputs' matrix) against the (100 x 1 'labels' vector).
        'loss' is a (1 x 1 scalar)."""
        
        # Backward and optimize

        # zero_grad will zero out all the previous gradient store in the parameters, w/o this, the gradients will be added to the old gradients
        optimizer.zero_grad()
        # perform backward backpropagation
        loss.backward()
        # update the parameters with radient descent
        optimizer.step()
        
        if (i+1) % 100 == 0:    # train_loader has 600 data so this will print 6 times in every epoch
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
            max_scores, predicted = torch.max(outputs.data, 1)
            print(f'Predictied:\n{predicted}')
            print(f'Labels:\n{labels}')

            total = labels.size(0)
            correct = 0
            correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the 100 images: {} %'.format(100 * correct / total))

            print('==========================================================')
            losses.append(loss.item())

max_scores, predicted = torch.max(outputs.data, 1)
# print('Final outputs:\n{}' .format(outputs))
# print('Final outputs.data\n{}' .format(outputs.data))
# print('Final max_scores\n{}' .format(max_scores))
print('Final predicted:\n{}' .format(predicted))
print('Final labels:\n{}' .format(labels))

# 5. Plot the loss curve

plt.plot(losses)

# 6. Test the network

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        """test_loader has 100 data. In each of the 100 data, the 'images' pack 
        contains 100 images, while the 'labels' pack also contains 100 labels. 
        Total there are 100 X 100 = 10,000 images and 10,000 respective labels"""
        images = images.reshape(-1, 28*28).to(device)   # train_loader on CPU, so send it to GPU
        labels = labels.to(device)   # train_loader on CPU, so send it to GPU
        outputs = model(images)
        max_scores, predicted = torch.max(outputs.data, 1)
        # print(max_scores, predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

print('Final predictions {}' .format(predicted))
print('Final labels {}' .format(labels))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.pt')
