import torch.cuda
from dataset import Mnist as Mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import model
from torch.utils.tensorboard import SummaryWriter


#使用平均池化层还是最大池化层,Max或者Avg
kind = 'Avg'
#数据集
trans = transforms.Compose([ transforms.ToTensor(), transforms.Resize((32, 32))])
train_dataset = Mnist(train=True, transforms=trans)
test_dataset = Mnist(train=False, transforms=trans)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#训练参数
epoch = 300
learning_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#模型,损失和优化器
if kind == 'Max':
    net = model.LeNet_Max().to(device)
else:
    net = model.LeNet_Avg().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

writer = SummaryWriter(f'LeNet_logs/{kind}')
best_accuracy = 0
best_epoch = 0
for i in range(epoch):
    print(f'epoch:{i}=====================================')
    net.train()
    sum_loss = 0
    for step, data in enumerate(train_dataloader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss
        rate = (step+1)/len(train_dataloader)
        a = int(rate * 50) * '*'
        b = (50-int(rate * 50)) * '_'
        print(f'\r[sum_loss:]{a}{b}——{sum_loss}', end='')
    print()
    writer.add_scalar('loss', sum_loss, i)

    with torch.no_grad():
        net.eval()
        sum_correct = 0
        for step, data in enumerate(test_dataloader):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)

            # 这里还可以使用torch.max(input, dim)函数来实现 (torch.max(outputs, dim=1)[1] == targets).sum().item()
            correct = (outputs.argmax(1) == targets).sum().item()
            sum_correct += correct
            accuracy = sum_correct/len(test_dataset)

            rate = (step + 1) / len(test_dataloader)
            a = int(rate * 50) * '*'
            b = (50 - int(rate * 50)) * '_'
            print(f'\r[accuracy:]{a}{b}——{accuracy}', end='')
        print()

        # 保存最高准确率以及对于模型权重
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = i
            torch.save(net.state_dict(), f'{kind}_best_accuracy_model.pth')
        # 保存最新模型权重
        torch.save(net.state_dict(), f'{kind}_later_model.pth')

        writer.add_scalar('accuracy on test', accuracy, i)
print(f'{kind}_best_accuracy:{best_accuracy}')
print(f'{kind}_best_epoch:{best_epoch}')
writer.close()




