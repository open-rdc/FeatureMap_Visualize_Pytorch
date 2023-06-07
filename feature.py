from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np

class FM_visualize:
    def __init__(self, module_name, layer_index):
        self.hook = module_name[layer_index].register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

        
parser = argparse.ArgumentParser(description='Feature Map Visualizing')
parser.add_argument('--img_path', default='./dog.jpg', type=str, help='image path')
args = parser.parse_args()


model = models.resnet18(pretrained=True).cuda()
print('Check the module name and number of the model!!')
print(model)

visual1 = FM_visualize(model.layer1, 0)   # Enter the model module and number to visualize
visual2 = FM_visualize(model.layer2, 0)   # Enter the model module and number to visualize
visual3 = FM_visualize(model.layer3, 0)   # Enter the model module and number to visualize
visual4 = FM_visualize(model.layer4, 0)   # Enter the model module and number to visualize

img = Image.open(args.img_path)
trans = transforms.ToTensor()
img = trans(img).unsqueeze(0).cuda()
model(img)

activations1 = visual1.features
feature1 = np.average(activations1[0],axis=0)
feature1z = np.pad(feature1, (2,1), 'constant')
plt.imshow(feature1)
plt.savefig('feature1.jpg')

activations2 = visual2.features
feature2 = np.average(activations2[0],axis=0)
feature2z = np.pad(feature2.repeat(2, axis=0).repeat(2, axis=1), (1,1), 'constant')
plt.imshow(feature2)
plt.savefig('feature2.jpg')

activations3 = visual3.features
feature3 = np.average(activations3[0],axis=0)
feature3z = feature3.repeat(4, axis=0).repeat(4, axis=1)
plt.imshow(feature3)
plt.savefig('feature3.jpg')

activations4 = visual4.features
feature4 = np.average(activations4[0],axis=0)
feature4z = feature3.repeat(4, axis=0).repeat(4, axis=1)
plt.imshow(feature4)
plt.savefig('feature4.jpg')

feature = np.multiply(feature1z, feature2z)
feature = np.multiply(feature, feature3z)
feature = np.multiply(feature, feature4z)
feature = (feature - feature.min())/(feature.max() - feature.min())
#feature = [[min(feature[i][j], 0.1) for j in range(144)] for i in range(144)]
feature = np.log((feature * 0.9)+0.1)
plt.imshow(feature)
plt.savefig('feature.jpg')

