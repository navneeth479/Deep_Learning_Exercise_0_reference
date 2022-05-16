import os
import json
import itertools
import scipy.misc
import numpy as np
import random
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from skimage.transform import resize


def cycle_mod(iterable):
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)
        for element in saved:
            yield element

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        
        self.file_names = os.listdir(file_path)
        self.file_names.sort(key = lambda x : int(x.split(".")[0]))
        
        if self.shuffle:
            self.iterator_gen = cycle_mod(self.file_names)
        else:
            self.iterator_gen = itertools.cycle(self.file_names)
        self.next_request_count = 0


        with open(self.label_path, 'r') as j:
            self.label_json = json.loads(j.read())
        
    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        
        image_names = [next(self.iterator_gen) for _ in range(self.batch_size)]
                
        if self.shuffle:
            random.shuffle(image_names)
            
        images = [self.augment(resize(np.load(self.file_path + i), self.image_size)) for i in image_names]
        
        labels = [int(i.split(".")[0]) for i in image_names]
        
        self.next_request_count += 1

        return (np.array(images), labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.rotation:
            img = np.rot90(img, k = random.choice([0, 1, 2, 3]))
            
        if self.mirroring:
            img = np.fliplr(img)
            
        return img

    def current_epoch(self):
        # return the current epoch number
        print(self.next_request_count)
        return self.next_request_count * self.batch_size // (len(self.file_names)+1)

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[self.label_json[x]]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        fig,axs=plt.subplots(4,3,figsize=(7,7))
        for (x,y,ax) in zip(images,labels,axs.flatten()):
            ax.imshow(x)
            ax.set_title(self.class_name(str(y)))
            ax.axis("off")
        fig.tight_layout()
        plt.show()
#



