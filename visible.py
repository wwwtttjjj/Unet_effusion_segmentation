import torch
from unet import UNet
from PIL import Image
import numpy as np
label2mask = [100, 200, 255]
NumClass = 4
modelpath = './checkpoints/unet.pth'
img_file = 'test.jpg'

net = UNet(n_channels=3, n_classes=NumClass,)
net.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
net.eval()  # 开启网络测试模式

img = np.array(Image.open(img_file))
img = img / 255.
img = img.resize((400, 400))
print(img)

img = torch.from_numpy(img.transpose((2, 0, 1))).type(torch.FloatTensor)

with torch.no_grad():
    output = net(img.unsqueeze(0).cpu())# 使用网络对图像进行预测
    pred = np.array(torch.argmax(output, dim = 1))
for i in range(len(label2mask)):
    pred[pred == (i + 1)] = label2mask[i]
pred = np.uint8(np.squeeze(pred))
print(np.unique(pred))
Image.fromarray(pred).save('test.png')