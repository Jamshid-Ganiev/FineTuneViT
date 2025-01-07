from modules.data_setup import get_data_loaders
from modules.engine import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_data_loaders("CIFAR-10", batch_size=128, input_size=(224, 224))

# Load and modify EfficientNet
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
model = model.to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate
results, best_accuracy = train_and_evaluate(
    model, train_loader, test_loader, loss_fn, optimizer, num_epochs=11, device=device,
    save_path="./trained_models/best_model_cifar10.pth", results_path="./CIFAR_10_EfficientNetB0_results.json"
)

print(f"Best Test Accuracy: {best_accuracy:.2f}%")
