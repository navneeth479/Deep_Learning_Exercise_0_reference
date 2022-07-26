from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

#                 
class ChallengeDataset(Dataset):
    def __init__(self, df, mode, seed=42):
        self._transform_train = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std)])
        self.mode = mode
        self.df = df

        self.input_list = []
        self.sum_crack = 0
        self.sum_inactive = 0

#        with open(params['input_csv_path']) as csv_file:
#            reader = csv.DictReader(csv_file, delimiter=';')
#            for row in reader:
#                self.input_list.append(row)
#
#        if valid_split_ratio == 0:
#            self.train_list = self.input_list
#        else:
#            self.train_list, self.valid_list = train_test_split(
#                self.input_list, test_size=valid_split_ratio, random_state=seed)
#
#        for row in self.train_list:
#            self.sum_crack += int(row['crack'])
#            self.sum_inactive += int(row['inactive'])
#

    def __len__(self):

        return self.df.shape[0]
#        if self.mode==Mode.TRAIN:
#            return len(self.train_list)
#        elif self.mode==Mode.VALID:
#            return len(self.valid_list)

    def __getitem__(self, index):
#        if self.mode==Mode.TRAIN:
#            output_list = self.train_list
#        elif self.mode==Mode.VALID:
#            output_list = self.valid_list

        label = np.zeros((2), dtype=int)
#        label[0] = int(output_list[idx]['crack'])
#        label[1] = int(output_list[idx]['inactive'])

        # Reads images using files name available in the list
        image = imread(self.df.iloc[index]["filename"])
        image = gray2rgb(image)
        image = self._transform_train(image)
        label = torch.from_numpy(np.array([self.df.iloc[index]["crack"], self.df.iloc[index]["inactive"]]))
        return image, label


#    def pos_weight(self):
#        '''
#        Calculates a weight for positive examples for each class and returns it as a tensor
#        Only using the training set.
#        '''
#        w_crack = torch.tensor((len(self.train_list) - self.sum_crack) / (self.sum_crack + epsilon))
#        w_inactive = torch.tensor((len(self.train_list) - self.sum_inactive) / (self.sum_inactive + epsilon))
#        output_tensor = torch.zeros((2))
#        output_tensor[0] = w_crack
#        output_tensor[1] = w_inactive
#
#        return output_tensor


#
#def get_train_dataset(cfg_path, valid_split_ratio):
#    # since all the images are 300 * 300, we don't need resizing
#    trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomVerticalFlip(p=0.5),
#                                transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),
#                                transforms.Normalize(train_mean, train_std)])
#    return ChallengeDataset(cfg_path=cfg_path,
#                            valid_split_ratio=valid_split_ratio, transform=trans, mode=Mode.TRAIN)

# without augmentation
#def get_validation_dataset(cfg_path, valid_split_ratio):
#    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
#                                transforms.Normalize(train_mean, train_std)])
#    return ChallengeDataset(cfg_path=cfg_path,
#                            valid_split_ratio=valid_split_ratio, transform=trans, mode=Mode.VALID)
