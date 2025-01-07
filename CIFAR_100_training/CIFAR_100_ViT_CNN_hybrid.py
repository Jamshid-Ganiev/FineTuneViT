from modules.data_setup import get_data_loaders
from modules.engine import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Hybrid CNN + ViT model
class CNN_ViT(nn.Module):
    def __init__(self, num_classes=100):
        super(CNN_ViT, self).__init__()

        # CNN Backbone
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        cnn_feature_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # ViT Backbone
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit_feature_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()

        # Combined Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_feature_dim + vit_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        combined = torch.cat((cnn_features, vit_features), dim=1)
        return self.classifier(combined)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data_loaders("CIFAR-100", batch_size=64, input_size=(224, 224))

# create model
model = CNN_ViT(num_classes=100).to(device)

# Loss function, optimizer, and scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Train and evaluate
results, best_accuracy = train_and_evaluate(
    model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs=15, device=device,
    save_path="./trained_models/best_model_cnn_vit_cifar100.pth", results_path="./CIFAR_100_ViT_CNN_Hybrid_results.json"
)

print(f"Best Test Accuracy: {best_accuracy:.2f}%")
