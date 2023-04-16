import torch 
from options.train_options import TrainOptions
import toml
from data.data_processing import augmentations, DeblurDataset, get_dataloader, save_images_dataloader_unet
import torchvision
import os 


def main(opt):
    
    #dataloading 
    target_aug_dict = toml.load(opt.augmentations_target_toml)
    input_aug_dict = toml.load(opt.augmentations_input_toml)
    
    dataset = DeblurDataset(opt.dataroot, augmentations(input_aug_dict) , augmentations(target_aug_dict))  

    
    #split the dataset to train and val
    train_dataset = torch.utils.data.Subset(dataset, range(0, int(0.8*len(dataset))))
    val_dataset = torch.utils.data.Subset(dataset, range(int(0.8*len(dataset)), len(dataset)))
        
    #make a dataloader 
    train_dataloader = get_dataloader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    val_dataloader = get_dataloader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    
    
    for og, blur in train_dataloader:
        print(og.shape)
        print(blur.shape)
        break
        
    
    os.makedirs(opt.model_outputs, exist_ok=True) 
        
    #save the images as a grid
    save_images_dataloader_unet(train_dataloader, os.path.join(opt.model_outputs, 'train.png'))
    
    
    #model import
    model = 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':

    opt = TrainOptions().parse()
    main(opt)
    
    
    
    