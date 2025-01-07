import torch
import os
import json

def train_and_evaluate(
    model, train_loader, test_loader, loss_fn, optimizer, scheduler=None, num_epochs=10, device="cpu",
    save_path=None, results_path=None, preprocess_fn=None
):
    """
    this function trains and evaluates a Pytorch model.
    args:
        model (torch.nn.Module): the model to train and evaluate.
        train_loader (DataLoader): for training data.
        test_loader (DataLoader): for testing data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        num_epochs (int): Number of epochs. Defaults to 10.
        device (str): Device for computation ('cpu' or 'cuda'). Defaults to 'cpu'.
        save_path (str, optional): Path to save the best model. Defaults to None.
        results_path (str, optional): Path to save results. Defaults to None.
        preprocess_fn (function, optional): Preprocessing function to apply during training and evaluation. Defaults to None.
    returns:
        dict: Training and evaluation results.
        float: Best accuracy achieved.
    """
    best_accuracy = 0.0
    results = {}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Preprocess images if needed
            if preprocess_fn:
                images = preprocess_fn(images)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Step the scheduler
        if scheduler:
            scheduler.step()

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                if preprocess_fn:
                    images = preprocess_fn(images)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total

        # Save the best model
        if save_path and test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)

        # Log epoch results
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%"
        )

        results[epoch + 1] = {
            "loss": avg_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }

    # Save results to JSON
    if results_path:
        os.makedirs(results_path, exist_ok=True)
        results["best_accuracy"] = best_accuracy
        with open(os.path.join(results_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

    return results, best_accuracy
