import torch 
from options.train_options import TrainOptions
import toml
from data.data_processing import augmentations, DeblurDataset, get_dataloader, save_images_dataloader_unet
import torchvision
import os 
from models.unet import UNet


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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #model import
    model = UNet(3,3)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss = torch.nn.MSELoss()
    
    mean_train_loss = 0
    mean_val_loss = 0
    
    train_losses = []
    val_losses = []
    

    #model training
    for epoch in range(opt.n_epochs):
        
        train_loss = 0
        val_loss = 0
        
        model.train()
        for i, (og, blur) in enumerate(train_dataloader):
            og = og.to(device)
            blur = blur.to(device)
            
            optimizer.zero_grad()
            
            output = model(blur)
            
            l = loss(output, og)
            
            l.backward()
            
            optimizer.step()
            
            print(f'Epoch {epoch} Batch {i} Loss {l.item()}')
            
            train_loss += l.item()
            
            #save the the og, blur and output images as a grid
            if i % 100 == 0:
                #stack the images
                images = torch.stack([og, blur, output])
                
                #save the images as a grid
                torchvision.utils.save_image(images, os.path.join(opt.model_outputs, f'epoch_{epoch}_batch_{i}.png'), nrow=3)
            

            
        #add no grad for validation
        model.eval()
        for i, (og, blur) in enumerate(val_dataloader):
            og = og.to(device)
            blur = blur.to(device)
            
            output = model(blur)
            
            l = loss(output, og)
            
            val_loss = l.item()
            
            print(f'Epoch {epoch} Validation Loss {l.item()}')
            
            if i % 50 == 0:
                #stack the images
                images = torch.stack([og, blur, output])
                
                #save the images as a grid
                torchvision.utils.save_image(images, os.path.join(opt.model_outputs, f'epoch_{epoch}_batch_{i}_val.png'), nrow=3)
  
        #mean loss
        mean_train_loss = train_loss / len(train_dataloader)
        mean_val_loss = val_loss / len(val_dataloader)
        
        train_losses = train_losses.append(mean_train_loss)
        val_losses = val_losses.append(mean_val_loss)
        
        

    
if __name__ == '__main__':

    opt = TrainOptions().parse()
    main(opt)
    
    
    
    