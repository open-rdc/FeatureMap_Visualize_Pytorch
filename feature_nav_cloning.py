import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.ndimage import convolve

class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # <Network CNN 3 + FC 2>
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)
    # <Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        # torch.nn.init.kaiming_normal_(self.fc5.weight)
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        self.flatten = nn.Flatten()
    # <CNN layer>
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            # self.maxpool,
            self.flatten
        )
    # <FC layer (output)>
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,

        )

    # <forward layer>
    def forward(self, x):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        return x2

class FM_visualize:
    def __init__(self, module_name, layer_index):
        self.hook = module_name[layer_index].register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

        
parser = argparse.ArgumentParser(description='Feature Map Visualizing')
parser.add_argument('--img_path', default='./frame.jpg', type=str, help='image path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3, 1)
model.to(device)
model.load_state_dict(torch.load("model_gpu.pt"))
#model = models.resnet18(pretrained=True).cuda()
print('Check the module name and number of the model!!')
print(model)

visual1 = FM_visualize(model.cnn_layer, 0)   # Enter the model module and number to visualize
visual2 = FM_visualize(model.cnn_layer, 2)   # Enter the model module and number to visualize
visual3 = FM_visualize(model.cnn_layer, 4)   # Enter the model module and number to visualize

img = Image.open(args.img_path)
trans = transforms.ToTensor()
img_cuda = trans(img).unsqueeze(0).cuda()
model(img_cuda)

activations1 = visual1.features
feature1 = np.average(activations1[0],axis=0)
plt.imshow(feature1)
plt.savefig('feature1.jpg')

activations2 = visual2.features
feature2 = np.average(activations2[0],axis=0)
plt.imshow(feature2)
plt.savefig('feature2.jpg')

activations3 = visual3.features
feature3 = -np.average(activations3[0],axis=0)
plt.imshow(feature3)
plt.savefig('feature3.jpg')

feature3z = np.pad(feature3, (1,1), 'edge')
filter3x3 = np.ones((3, 3))
feature3d = convolve(feature3z, filter3x3)
feature2m = np.multiply(feature3d, feature2)
feature2z = np.pad(feature2m.repeat(2, axis=0).repeat(2, axis=1), (1,1), 'edge')
feature2d = convolve(feature2z, filter3x3)
feature1z = np.pad(feature1, (0,1), 'edge')
feature0 = np.multiply(feature2d, feature1z)
filter8x8 = np.ones((8, 8))
feature0z = feature0.repeat(4, axis=0).repeat(4, axis=1)
feature = convolve(feature0z, filter8x8)

feature = (feature - feature.min())/(feature.max() - feature.min())
#feature = np.log10((feature * 0.9)+0.1)+1
plt.imshow(feature)
plt.savefig('feature.jpg')

#feature_rgb = np.repeat(feature[:, :, np.newaxis], 3, axis=2)
a = np.ones(feature.shape)
feature_rgb = np.dstack((feature, feature, feature))
img_extract = np.multiply(feature_rgb, img)

img_extract = (img_extract - img_extract.min())/(img_extract.max() - img_extract.min())
plt.imshow(img_extract)
plt.savefig('extract_image.jpg')

