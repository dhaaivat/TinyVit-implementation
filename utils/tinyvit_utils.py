
#imports
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


#Trying it out on CIFAR-10
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4942, 0.4851, 0.4504],
                         std=[0.2467, 0.2429, 0.2616])
])



def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loop = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}%")

        # ---------- Validation ----------
        model.eval()
        val_correct, val_total = 0, 0

        val_loop = tqdm(val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # ---------- Track Best ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_tinyvit_cifar10.pth")
            print("Best model saved!")

    print(f"\n Best Validation Accuracy: {best_val_acc:.2f}%")
