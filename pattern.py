import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self,resolution,tile_size):
        self.resolution=resolution
        self.tile_size=tile_size
        self.output=None
        self.no_black_while_tiles=int(self.resolution/(2*self.tile_size))

    def draw(self):
        zeros=np.zeros(self.tile_size,dtype=int)
        ones=np.ones(self.tile_size,dtype=int)
        #0011
        zo=np.concatenate((zeros,ones),axis=None)
        #001100110011
        x_zo = np.tile(zo, self.no_black_while_tiles)
        #1100
        oz=np.concatenate((ones,zeros),axis=None)
        #11001100
        x_oz = np.tile(oz, self.no_black_while_tiles)
        tile_row1=np.tile(x_zo,(self.tile_size,1))
        tile_row2 = np.tile(x_oz, (self.tile_size, 1))
        tile_black_white=np.concatenate((tile_row1,tile_row2),axis=0)
        #create the output matrix by repeating the first 2 rows no_black_while_tiles times
        self.output=np.tile(tile_black_white,(self.no_black_while_tiles,1))
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output,cmap='gray')
        plt.show()