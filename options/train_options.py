from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        #dataset root path
        self.parser.add_argument('--dataroot', type=str, default='/home/saiteja/detectwork/helmetdetection/completedataset/helmet_classification/helmet', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--root1', type=str, default='/home/saiteja/detectwork/helmetdetection/completedataset/helmet_classification/helmet', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--root2', type=str, default='/home/saiteja/detectwork/helmetdetection/completedataset/helmet_classification/helmet', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        
        #model name
        self.parser.add_argument('--modelname', type=str, default='resnet18', help='name of the experiment. It decides where to store samples and models')

        #model_outputs
        self.parser.add_argument('--model_outputs', type=str, default='model_outputs', help='name of the experiment. It decides where to store samples and models')

        #no of classes
        self.parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
        
        #augmentation path
        self.parser.add_argument('--augmentations_target_toml', type=str, default='metadata/augmentation_target.toml', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        self.parser.add_argument('--augmentations_input_toml', type=str, default='metadata/augmentation_input.toml', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        # batch_size=16, shuffle=True, num_workers=4, pin_memory=True
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the dataset')
        self.parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
        self.parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
        
        #lr
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        
        #n_epochs
        self.parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')

        self.isTrain = True
