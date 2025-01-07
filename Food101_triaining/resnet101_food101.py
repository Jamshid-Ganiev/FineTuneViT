from modules.data_setup import get_data_loaders
from modules.engine import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_data_loaders("Food-101", batch_size=256, input_size=(224, 224))

# Load and modify ResNet-101
model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 101)  #101 classes
)
model = model.to(device)

# Loss function, optimizer, and scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Train and evaluate
results, best_accuracy = train_and_evaluate(
    model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs=50, device=device,
    save_path="./trained_models/best_model_resnet101_food101.pth", results_path="./Food101_Resnet101_full_fine_tuning_results.json"
)

print(f"Best Test Accuracy: {best_accuracy:.2f}%")
