import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image, ImageOps
from torchvision.utils import save_image


#PARAMS
model_path = 'mnist_vdautoencoder.pth'
image_path = 'data/MNIST/sample_images/' 
output_path = 'gifs/VAE_smallsample/'
step = 7       #fetch everything nth image, (bigger numbers mean less mnist digits in final combined image)

if not os.path.exists(output_path):
	os.mkdir(output_path)

#model
from models import autoencoder, VariationalAutoencoder
model = VariationalAutoencoder()
model.load_state_dict(torch.load(model_path))
model.cuda()


#Image Loading

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 200
batch_size = 128
learning_rate = 1e-3


def add_noise(img):
    noise = torch.randn(img.size()) * 0.4
    noisy_img = img + noise
    return noisy_img


def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)



image_names = os.listdir(image_path)
image_names = sorted(image_names)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])

totens= transforms.Compose([
    transforms.ToTensor()])

def image_loader(image_folder = image_path, transform=img_transform,step = step ):
	img_names = os.listdir(image_folder)
	img_names.sort()
	image_bw_list = []
	image_gs_list = []
	for i,img_name in enumerate(img_names):
		if i%step != 0:
			continue
		image = Image.open(os.path.join(image_folder,img_name))
		#black and white version of inputs
		image_bw = transform(image).float()
		image_bw = image_bw.unsqueeze(0)
		image_bw = image_bw.cuda()
		image_bw_list.append(image_bw)
		#grayscale version of inputs
		image_gs = totens(image).float()
		image_gs = image_gs.unsqueeze(0)
		image_gs = image_gs.cuda()
		image_gs_list.append(image_gs)
	return (torch.cat(image_bw_list,0),torch.cat(image_gs_list,0))

#format imput
images_bw,images_gs = image_loader()
images_bw,images_gs = images_bw.view(images_bw.size(0), -1),images_gs.view(images_gs.size(0), -1)
images_bw,images_gs = Variable(images_bw).cuda(),Variable(images_gs).cuda()

#run model
images = images_gs
for i in range(100):
	pic = to_img(images.cpu().data)
	save_image(pic, '%s/pic_%s.png'%(output_path,i))
	images, mu, logvar = model(images)      #GET RID OF mu and logvar for non variational autoencoder


'''
inpic_bw,inpic_gs = to_img(images_bw.cpu().data),to_img(images_gs.cpu().data)
outpic_bw,outpic_gs = to_img(output_bw.cpu().data),to_img(output_gs.cpu().data)
save_image(inpic_bw, './gifs/in_bw.png')
save_image(inpic_gs, './gifs/in_gs.png')
save_image(outpic_bw, './gifs/out_bw.png')
save_image(outpic_gs, './gifs/out_gs.png')
'''

'''
#convert to GIF
import subprocess
import os
 
i = output_path+"*.png"
o = output_path+"output.gif"
subprocess.call("convert -delay 100 -loop 0 " + i + " " + o, shell=True)
'''
#os.system("start output.gif")
