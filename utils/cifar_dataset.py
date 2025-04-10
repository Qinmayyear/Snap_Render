import torchvision
import torchvision.transforms as transforms

# 设置图像变换（可能需要根据模型输入尺寸调整）
transform = transforms.Compose([
    transforms.Resize((32, 32)),        # 可选，CIFAR-10 本身就是 32x32
    transforms.ToTensor()
])

# 下载和加载 CIFAR-10 训练和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)