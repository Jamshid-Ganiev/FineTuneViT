from modules.data_setup import get_data_loaders
from modules.engine import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_data_loaders("FashionMNIST", batch_size=128, input_size=(28, 28))

# create a model
model = SimpleCNN().to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate
results, best_accuracy = train_and_evaluate(
    model, train_loader, test_loader, loss_fn, optimizer, num_epochs=25, device=device,
    save_path="./trained_models/best_cnn_model.pth", results_path="./FashionMNIST_CNN_results.json"
)

print(f"Best Test Accuracy: {best_accuracy:.2f}%")
