import torch
import model
from PIL import Image
import torchvision.transforms as transforms

#准备图片
path = 'Image/8.jpg'
img = Image.open(path).convert('L')
trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
img = trans(img)
transforms.ToPILImage()(img).show()
img = img.reshape((1, 1, 32, 32))



#显示图像
#准备模型
#模型类型，Max或Avg
kind = 'Max'

if kind == 'Max':
    net = model.LeNet_Max()
else:
    net = model.LeNet_Avg()

net.load_state_dict(torch.load(f'{kind}_best_accuracy_model.pth'))

#检测
output = torch.max(net(img),dim=1)[1].item()
print(output)