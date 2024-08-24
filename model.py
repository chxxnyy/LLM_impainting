import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.models.vgg as vgg
import torchvision.transforms.functional as F



# 모델의 층을 초기화 시킬 때 쓴 코드로, 필요하지 않으시다면 사용하지 않으셔도 됩니다.
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2

    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]

    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight).float()


class segmentation_model(nn.Module):
    def __init__(self, n_class=7):
        super(segmentation_model, self).__init__()
        self.features = models.vgg16(pretrained=True).features

        # fc6 layer
        self.fc6 = nn.Conv2d(512, 4096, 7)


        # fc6 layer
        self.fc7 = nn.Conv2d(4096, 4096, 1)


        # ReLU
        self.relu = nn.ReLU(inplace=True)

        # Dropout
        self.dropout = nn.Dropout2d()


        # 각 단계에 대한 prediction convolutions (Downsampling) 을 정의
        self.predict_conv1 = nn.Conv2d(4096, n_class, kernel_size=1)
        self.predict_conv2 = nn.Conv2d(512, n_class, kernel_size=1)
        self.predict_conv3 = nn.Conv2d(256, n_class, kernel_size=1)

        # Upscaling prediction (Upsampling)을 위한 Deconvolution layer를 정의
        self.deconv1 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, bias=False)
        self.deconv2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, bias=False)
        self.deconv3 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, bias=False)

        self._initialize_weights()
        self.copy_params_from_vgg16()

        
        

    def _initialize_weights(self):
        # [2] 빈칸을 작성하시오.
        self.features[0].padding = (50,50)

        for m in self.modules():
            if isinstance(m, nn.MaxPool2d):
                m.ceil_mode = True
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def forward(self, x):
        initial = x
        # VGG 네트워크를 통과시키고, pool3와 pool4의 결과를 저장한다.
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 16:
                pool3 = x.clone()
            elif idx == 23:
                pool4 = x.clone()

        # Pass through fc6 layer
        x = self.dropout(self.relu(self.fc6(x)))
        print(x.shape)

        # Pass through fc7 layer
        x = self.dropout(self.relu(self.fc7(x)))

        # Pass through Prediction 1 and Deconvolution 1
        x = self.predict_conv1(x)
        x = self.deconv1(x)
        deconv1 = x

        # Pass through Prediction 2
        x = self.predict_conv2(pool4)
        x = x[:, :, 5:5 + deconv1.size(2), 5:5 + deconv1.size(3)] # Crop boundary
        predict2 = x

        # Add deconv1 and predict2, then pass through Deconvolution 2
        x = deconv1 + predict2 * 0.01
        x = self.deconv2(x)
        deconv2 = x

        # Pass through Prediction 3
        x = self.predict_conv3(pool3)
        x = x[:, :, 9:9 + deconv2.size(2), 9:9 + deconv2.size(3)] # Crop boundary
        predict3 = x

        # Add predict3 and deconv2
        x = predict3 * 0.0001 + deconv2

        # Pass through Deconvolution 3
        x = self.deconv3(x)
        x = x[:, :, 31:31+initial.size(2), 31:31+initial.size(3)].contiguous() # Crop boundary
        return x

    def copy_params_from_vgg16(self): # pre-train된 VGG16 모델의 매개변수를 복사하는 함수입니다. 조금 어려울 수 있으니 이해하지 않고 넘어가셔도 됩니다.
        vgg16 = models.vgg16(pretrained=True)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
            
            

x = np.random.rand(1, 128, 128)
model = segmentation_model()
model(x)
