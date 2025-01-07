# FineTuneViT
Image classification using Vision Transformers (ViTs) and Transfer Learning across CIFAR, Fashion-MNIST, and Food-101 datasets.

[View the Report](./Report_pdf.pdf)


# Vision Transformers and Transfer Learning for Image Classification

This project explores the integration of **Vision Transformers (ViTs)** with **Transfer Learning** to perform image classification across multiple datasets, including:
- CIFAR-10
- CIFAR-100
- Fashion-MNIST
- Food-101

## Key Features
- Fine-tuning of pre-trained ViT models on specific datasets.
- Hybrid architectures combining ViT and ResNet-18 for enhanced performance.
- Data augmentation techniques: resizing, cropping, flipping, and normalization.
- Optimization using cross-entropy loss and the Adam optimizer.
- Reduced training time with Transfer Learning.

## Highlights
- Achieved 90%+ accuracy on CIFAR-10, CIFAR-100, and Fashion-MNIST datasets.
- Achieved 85%+ accuracy on Food-101 with fully fine-tuned ResNet-101.
- Demonstrated training time reduction from 9 hours to 6 hours using Transfer Learning.

## Applications
This project showcases the potential of Vision Transformers for real-world applications, including medical image analysis, where identifying subtle patterns in X-rays, CT scans, and MRI images is critical.

## Datasets and Pre-trained Models
- **Datasets**: CIFAR-10, CIFAR-100, Fashion-MNIST, Food-101
- **Models**: EfficientNet-B0, Vision Transformer (ViT-B/16), ResNet-18, ResNet-101

