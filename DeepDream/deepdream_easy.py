# the original code is from: https://github.com/wmn7/ML_Practice/blob/master/2019_05_27/deep-dream-pytorch.ipynb
# references: https://mathpretty.com/10485.html, https://mathpretty.com/10475.html

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageChops
import torch
import torch.optim as optim
from torchvision import models, transforms


# Configs
####################################################################################################
CUDA_ENABLED = False
LAYER_ID = 16  # The deepest activated layer, default=28 （for VGG16, LAYER_ID<=31）
NUM_ITERATIONS = 3  # Number of iterations to update the input image with the layer's gradient, default=3
NUM_DOWNSCALES = 15  # downscale levels, default=20
BLEND_ALPHA = 0.5  # ratio between the blurred small image and the original image


class DeepDream(object):

    def __init__(self, image):
        self.image = image
        self.img_size = 224  # initial image size for pretrained network, e.g., vgg16 use 224x224 images
        self.lr = 0.2  # learning rate
        transform_mean = [0.485, 0.456, 0.406]  # initial mean and std of ImageNet to normalize images
        transform_std = [0.229, 0.224, 0.225]
        self.transform_pre = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std)
        ])
        self.tensor_mean = torch.Tensor(transform_mean)
        self.tensor_std = torch.Tensor(transform_std)
        self.model = models.vgg16(pretrained=True)  # load pre-trained model VGG116
        if CUDA_ENABLED:
            self.model = self.model.cuda()
            self.tensor_mean = self.tensor_mean.cuda()
            self.tensor_std = self.tensor_std.cuda()
        self.modules = list(self.model.features.modules())

    def to_image(self, image):
        return image * self.tensor_std + self.tensor_mean

    def dream_core(self, image, layer, iterations):
        # set image
        image = self.transform_pre(image).unsqueeze(0)
        if CUDA_ENABLED:
            image = image.cuda()
        image = torch.autograd.Variable(image, requires_grad=True)
        # update parameters
        self.model.zero_grad()
        optimizer = optim.Adam([image.requires_grad_()], lr=self.lr)
        for _ in range(iterations):
            optimizer.zero_grad()
            out = image
            for layer_id in range(layer):
                out = self.modules[layer_id + 1](out)
            loss = -out.norm()  # 让负的变小, 正的变大
            loss.backward()
            optimizer.step()
            # image.data = image.data + lr * image.grad.data
        # recover image
        image = image.data.squeeze()
        image.transpose_(0, 1)
        image.transpose_(1, 2)
        image = self.to_image(image)
        if CUDA_ENABLED:
            image = image.cpu()
        image = np.clip(image, 0, 1)
        image = Image.fromarray(np.uint8(image * 255))
        return image

    def dream_recursive(self, image, layer, iterations, num_downscales):
        if num_downscales > 0:
            # scale down the input image
            image_small = image.filter(ImageFilter.GaussianBlur(2))
            small_size = (int(image.size[0] / 2), int(image.size[1] / 2))
            if small_size[0] == 0 or small_size[1] == 0:
                small_size = image.size
            image_small = image_small.resize(small_size, Image.ANTIALIAS)
            # run dream_recursive on the scaled-down image
            image_small = self.dream_recursive(image_small, layer, iterations, num_downscales - 1)
            # scale up the result image to the original size
            image_large = image_small.resize(image.size, Image.ANTIALIAS)
            # blend the two image
            image = ImageChops.blend(image, image_large, BLEND_ALPHA)
            print('now num_downscalses = {} ...'.format(num_downscales))
        # update gradient by the current image in order to amplify its features through neural network
        img_result = self.dream_core(image, layer, iterations)
        img_result = img_result.resize(image.size)
        print(img_result.size)
        return img_result

    def run_dream(self):
        return self.dream_recursive(self.image, LAYER_ID, NUM_ITERATIONS, NUM_DOWNSCALES)


if __name__ == "__main__":
    load_path = "your_image.jpg"
    img_dream = DeepDream(Image.open(load_path)).run_dream()
    save_path = "dreamed_"+load_path
    plt.imshow(img_dream)
    plt.savefig(save_path)
