import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self,resolution,tile_size):
        self.resolution=resolution
        self.tile_size=tile_size
        self.output=None
        self.no_black_while_tiles=int(self.resolution/(2*self.tile_size))
        if (self.resolution%(2*self.tile_size))!=0:
            raise AssertionError("Resolution is not evenly divisible")

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
        
class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.cx, self.cy = position[0], position[1]
        
    
    def draw(self):
        x = np.linspace(0, self.resolution, self.resolution)
        y = np.linspace(0, self.resolution, self.resolution)
        
        xx, yy = np.meshgrid(x, y)
        
        mask = (xx - self.cx)**2+(yy - self.cy)**2-self.radius**2 <= 0
#        mask = mask.astype(int)

        self.output = mask
        return mask.copy()
        
    def show(self):
        plt.imshow(self.output)
        


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
    
    def draw(self):
        op=np.zeros([self.resolution,self.resolution, 3])
        op[:,:,0]= np.linspace(0,1,self.resolution)
        op[:,:,1]=np.linspace(0,1,self.resolution).reshape (self.resolution, 1)
        op[:,:,2]= np.linspace(1,0,self.resolution)
        self.output = op
        return op.copy()
    
    def show(self):
        plt.imshow(self.output)