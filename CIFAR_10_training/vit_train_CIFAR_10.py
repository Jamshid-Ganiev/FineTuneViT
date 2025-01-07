from modules.data_setup import get_data_loaders
from modules.engine import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_data_loaders("CIFAR-10", batch_size=128, input_size=(224, 224))

# Load and modify ViT
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
model.heads = nn.Linear(model.heads[0].in_features, 10)
model = model.to(device)

# Freeze all layers except the last two encoder layers
for name, param in model.named_parameters():
    if "encoder.layers.10" in name or "encoder.layers.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Train and evaluate
results, best_accuracy = train_and_evaluate(
    model, train_loader, test_loader, loss_fn, optimizer, num_epochs=11, device=device,
    save_path="./trained_models/best_model_vit_cifar10.pth", results_path="./CIFAR_10_VIT_results.json"
)

print(f"Best Test Accuracy: {best_accuracy:.2f}%")