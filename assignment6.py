import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

for i in range(len(train_dataset)):
    img, label = train_dataset[i]
    assert img.shape == (1, 28, 28), f"Invalid dimensions for training image {i+1}"
    assert img.min() >= 0 and img.max() <= 1, f"Invalid pixel values for training image {i+1}"

for i in range(len(test_dataset)):
    img, label = test_dataset[i]
    assert img.shape == (1, 28, 28), f"Invalid dimensions for test image {i+1}"
    assert img.min() >= 0 and img.max() <= 1, f"Invalid pixel values for test image {i+1}"

class ComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

input_size = 28 * 28
hidden_size1 = 500
hidden_size2 = 300
output_size = 10

model = ComplexNN(input_size, hidden_size1, hidden_size2, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy * 100:.2f}%')

print(f'\nParameters:')
print(f'Input Size: {input_size}')
print(f'Hidden Size 1: {hidden_size1}')
print(f'Hidden Size 2: {hidden_size2}')
print(f'Output Size: {output_size}')
print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
print(f'Weight Decay (L2 Regularization): {optimizer.param_groups[0]["weight_decay"]}')
print(f'Batch Size: {batch_size}')