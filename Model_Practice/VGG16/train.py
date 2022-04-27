import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import vgg
from tqdm import tqdm

data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'val':transforms.Compose([transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

train_dataset = ImageFolder('dataset/train', data_transform['train'])
test_dataset = ImageFolder('dataset/test', data_transform['val'])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

epoch = 100
learning_rate = 0.00001
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net = vgg('vgg16', class_num=5).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
best_acc = 0.0
for epoch_idex in range(epoch):
    net.train()
    sum_loss = 0
    tbar = tqdm(train_dataloader, desc=f'epoch:{epoch_idex}')
    for step, data in enumerate(tbar):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        tbar.set_postfix(sum_loss=sum_loss)

    with torch.no_grad():
        net.eval()
        correct = 0
        tbar = tqdm(test_dataloader, desc='test')
        for data in tbar:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = net(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()

            accuracy = correct/len(test_dataset)
            tbar.set_postfix(accuracy=accuracy)
        if best_acc < accuracy:
            best_acc = accuracy
            # 保存训练效果最好的模型
            torch.save(net.state_dict(), 'best_model.pth')
        print(f'accuracy:{accuracy}')


