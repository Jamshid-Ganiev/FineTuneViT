from modules.data_setup import get_data_loaders
from modules.engine import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_data_loaders("FashionMNIST", batch_size=256, input_size=(224, 224))

# Load and modify ViT
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
model.heads = nn.Linear(model.heads[0].in_features, 10)  # 10 classes
model = model.to(device)

# Freeze all layers except the classification head
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "heads" in name:
        param.requires_grad = True

# Grayscale-to-RGB preprocessing function
def preprocess_fn(images):
    return images.repeat(1, 3, 1, 1)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Train and evaluate
results, best_accuracy = train_and_evaluate(
    model, train_loader, test_loader, loss_fn, optimizer, num_epochs=15, device=device,
    save_path="./trained_models/best_vit_model_fashionmnist.pth", results_path="./FashionMNIST_ViT_results.json",
    preprocess_fn=preprocess_fn
)

print(f"Best Test Accuracy: {best_accuracy:.2f}%")
