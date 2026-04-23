import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image


idx2label = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}


class BottleneckBlock(nn.Module):
  def __init__(self,in_channels,growth_rate):
    super(BottleneckBlock,self).__init__()
    self.bn1=nn.BatchNorm2d(in_channels)
    self.conv1=nn.Conv2d(in_channels,4*growth_rate,kernel_size=1,bias=False)
    self.bn2=nn.BatchNorm2d(4*growth_rate)
    self.conv2=nn.Conv2d(4*growth_rate,growth_rate,kernel_size=3,padding=1,bias=False)
    self.relu=nn.ReLU()
  def forward(self,x):
    res=x
    x=self.bn1(x)
    x=self.relu(x)
    x=self.conv1(x)
    x=self.bn2(x)
    x=self.relu(x)
    x=self.conv2(x)
    x=torch.cat([res,x],1)

    return x


class DenseBlock(nn.Module):
  def __init__(self,num_layers,in_channels,growth_rate):
    super(DenseBlock,self).__init__()
    layers=[]
    for i in range(num_layers):
      layers.append(BottleneckBlock(in_channels+i*growth_rate,growth_rate))
    self.block=nn.Sequential(*layers)
  def forward(self,x):
    return self.block(x)



class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_classes):
        super(DenseNet, self).__init__()

        # 1. Lớp Convolution đầu tiên (Initial Convolution)
        # Output channels thường là 2 * growth_rate
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense_blocks = nn.ModuleList()
        in_channels = 2 * growth_rate

        # 2. Xây dựng các Dense Blocks và Transition Layers
        for i, num_layers in enumerate(num_blocks):
            # Thêm một Dense Block
            self.dense_blocks.append(DenseBlock(num_layers, in_channels, growth_rate))
            in_channels += num_layers * growth_rate

            # Nếu không phải block cuối cùng, thêm một Transition Layer để giảm số kênh và kích thước ảnh
            if i != len(num_blocks) - 1:
                out_channels = in_channels // 2
                self.dense_blocks.append(nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                ))
                in_channels = out_channels

        # 3. Lớp kết thúc (Final Layers)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # Sử dụng AdaptiveAvgPool2d để luôn đưa về kích thước 1x1 bất kể ảnh đầu vào lớn hay nhỏ
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Chạy qua danh sách ModuleList
        for block in self.dense_blocks:
            x = block(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Flatten dữ liệu cho lớp Linear
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def transform(img,img_size=(224,224)):
  img=img.resize(img_size)
  img=np.array(img)[...,:3]
  img=torch.tensor(img).permute(2,0,1).float()
  normalized_img=img/255.0
  return normalized_img

@st.cache_resource
def load_model(model_path):
    model=DenseNet(num_blocks=[6,12,24,16],growth_rate=32,num_classes=6)
    model.load_state_dict(torch.load(model_path,weights_only=True,map_location=torch.device('cpu')))
    model.eval()
    return model
model=load_model('DenseNet_model.pth')

def inference(image,model):
  img=transform(image)
  img=img.unsqueeze(0)
  with torch.no_grad():
    output=model(img)
    preds=nn.Softmax(dim=1)(output)
    p_max,yhat=torch.max(preds.data,1)
    return p_max.item()*100,yhat.item()

def main():
    st.title('Scene Classification')
    st.subheader('Model: DenseNet. Dataset: Scene Classification')
    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image", type=["jpg", "png"])
        if file is not None:
          image = Image.open(file)
          p, idx = inference(image, model)
          label = idx2label[idx]
          st.image(image,width=700)
          st.success(f"The uploaded image is of the {label} with {p:.2f} % probability.") 

    elif option == "Run Example Image":
      image = Image.open('demo_scene.jpg')
      p, idx = inference(image, model)
      label = idx2label[idx]
      st.image(image,width=700)
      st.success(f"The image is of the {label} with {p:.2f} % probability.") 

if __name__ == '__main__':
    main()