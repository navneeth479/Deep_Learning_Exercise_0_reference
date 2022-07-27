from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

#                 
class ChallengeDataset(Dataset):
    def __init__(self, df, mode = "any"):
        self._transform_train = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std)])
        self.mode = mode
        self.df = df

        self.input_list = []
        self.sum_crack = 0
        self.sum_inactive = 0


    def __len__(self):
        return self.df.shape[0]
    
    
    
    def __getitem__(self, index):
        image = imread(self.df.iloc[index]["filename"])
        image = gray2rgb(image)
        image = self._transform_train(image)
        label = torch.from_numpy(np.array([self.df.iloc[index]["crack"], self.df.iloc[index]["inactive"]]))
        return image, label
