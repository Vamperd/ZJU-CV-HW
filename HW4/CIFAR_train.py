import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from resnet import ResNet18


def get_device():
    print(f"PyTorch version: {torch.__version__}")
    import torchvision
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cutout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img
        
        img = np.array(img)
        h, w = img.shape[:2]
        length = h // 2
        
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        img[y1:y2, x1:x2] = 0
        
        return Image.fromarray(img)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4), # 调整 RandomCrop 参数适应 32x32
        Cutout(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Acc")
    plt.plot(epochs, test_accuracies, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curves")
    plt.legend()

    save_path = os.path.join(out_dir, "cifar_training_curves_small_input.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved curves to {save_path}")


def main():
    device = get_device()

    train_transform, test_transform = get_transforms()

    data_root = os.path.join(os.path.dirname(__file__), 'data')
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # 移除 Nesterov, Weight Decay 等优化
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 移除 Scheduler

    num_epochs = 65
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 30 == 0:
                print(f'Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_dataset)
        train_accuracy = 100 * correct_train / total_train
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Time: {elapsed_time:.2f}s')

        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss /= len(test_dataset)
        test_accuracy = 100 * correct_test / total_test
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

    # Save curves
    out_dir = os.path.join(os.path.dirname(__file__), 'result', 'ResNet')
    plot_curves(train_losses, test_losses, train_accuracies, test_accuracies, out_dir)


if __name__ == '__main__':
    main()
