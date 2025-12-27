import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from lenet5 import LeNet5


def main():
    # 设置设备
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available, using CPU")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载MNIST数据集
    try:
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            transform=transform, 
            download=True
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            transform=transform, 
            download=True
        )
    except RuntimeError as e:
        print(f"Error loading dataset: {e}")
        print("This might be due to missing numpy or PIL (Pillow)")
        print("Please ensure numpy and pillow are installed: pip install numpy pillow")
        return
    
    # 创建数据加载器
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # 初始化模型、损失函数和优化器
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 训练参数
    num_epochs = 10
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss_avg = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss_avg)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss_avg:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    # 绘制训练和测试曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('result/lenet5/mnist_training_curves.png')
    plt.show()
    
    # 最终测试准确率
    print(f'Final Test Accuracy: {test_accuracies[-1]:.2f}%')


if __name__ == "__main__":
    main()