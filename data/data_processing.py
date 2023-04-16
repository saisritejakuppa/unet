import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from logzero import logger
import numpy as np

def augmentations(aug_dict, default_prob=0.5):
    """Augmentations for the dataset.
    Args:
        aug_dict (dict): Dictionary containing the names of the augmentations as keys and their corresponding parameters.
        default_prob (float): Default probability value to use if not specified in the toml file.
    Returns:
        torchvision.transforms.Compose: Augmentations for the dataset.
    """
    custom_transforms = []

    # resize the images to 224x224
    custom_transforms.append(A.Resize(224, 224))

    # use albumentations for augmentations
    aug_map = {
        'blur': A.Blur,
        'RandomHorizontalFlip': A.HorizontalFlip,
        'RandomVerticalFlip': A.VerticalFlip,
        'RandomRotation': A.Rotate,
        'GaussianBlur': A.GaussianBlur,
        'ColorJitter': A.ColorJitter,
        'RandomCrop': A.RandomCrop,
        'RandomBrightnessContrast': A.RandomBrightnessContrast,
        'Cutout': A.Cutout,
        'RandomResizedCrop': A.RandomResizedCrop,
        'RandomRain': A.RandomRain,
        'RandomSnow': A.RandomSnow,
        'RandomSunFlare': A.RandomSunFlare,
        'RandomFog': A.RandomFog,
        'RandomSnow': A.RandomSnow,
        'Downscale': A.Downscale,
        'ImageCompression': A.ImageCompression,
    }

    for aug_name, aug_params in aug_dict.items():
        if aug_name in aug_map:
            aug_fn = aug_map[aug_name]
            aug_fn_args = {k: v for k, v in aug_params.items() if k not in ['p']}
            aug_fn_prob = aug_params.get('p', default_prob)
            # if aug_name == 'RandomSnow':
            #     aug_fn_args['value'] = aug_params.get('value', 50)
            custom_transforms.append(aug_fn(p=aug_fn_prob, **aug_fn_args))

    # normalize the images to cifar mean and std
    # custom_transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    custom_transforms.append(ToTensorV2())

    custom_transforms = A.Compose(custom_transforms)

    print(custom_transforms)
    
    
    return custom_transforms

def save_images_dataloader(custom_dataloader, imgname):
    """Save images from the dataloader.
    Args:
        custom_dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
        imgname (str): Name of the output image file.
    """
    images, _ = next(iter(custom_dataloader))  # get a batch of images from the dataloader
    grid = torchvision.utils.make_grid(images)  # create a grid of images

    # save the grid as an image file
    torchvision.utils.save_image(grid, imgname)


def save_images_dataloader_unet(custom_dataloader, imgname):
    """Save images from the dataloader.
    Args:
        custom_dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
        imgname (str): Name of the output image file.
    """
    input_imgs, target_imgs = next(iter(custom_dataloader))  # get a batch of images from the dataloader
    inp_grid = torchvision.utils.make_grid(input_imgs)  # create a grid of images
    tar_grid = torchvision.utils.make_grid(target_imgs)  # create a grid of images

    # save the grid as an image file
    torchvision.utils.save_image(inp_grid, imgname.replace('.png', '_input.png'))
    torchvision.utils.save_image(tar_grid, imgname.replace('.png', '_target.png'))





def get_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True):
    """
    Returns a PyTorch DataLoader for a given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


class DeblurDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, blur_transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.blur_transform = blur_transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.root_dir)[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        
        
        #convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image / 255.0
        
        #convert to float 64
        image = image.astype(np.float32)
        
        if self.transform:
            input_img = self.transform(image= image)['image']
        
        if self.blur_transform:
            target_img = self.blur_transform(image= image)['image']

        return input_img, target_img




class ImageMatchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.transform = transform

        # Get the list of image names in both directories
        self.img_names1 = sorted(os.listdir(root_dir1))
        self.img_names2 = sorted(os.listdir(root_dir2))

    def __len__(self):
        return len(self.img_names1)

    def __getitem__(self, idx):
        # Get the corresponding image names from both directories
        img_name1 = self.img_names1[idx]
        img_name2 = img_name1  # Assume the image names match

        if img_name2 != img_name1:
            print("Image names don't match: {} and {}".format(img_name1, img_name2))

        img_path1 = os.path.join(self.root_dir1, img_name1)
        img_path2 = os.path.join(self.root_dir2, img_name2)

        image1 = Image.open(img_path1)
        image2 = Image.open(img_path2)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2
