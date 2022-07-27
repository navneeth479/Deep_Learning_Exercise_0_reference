import torch as t
import numpy as np
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
        self.train_batch_sampler = t.utils.data.BatchSampler(t.utils.data.RandomSampler(range(len(self._train_dl))), 100, False)
        self.test_batch_sampler = t.utils.data.BatchSampler(t.utils.data.RandomSampler(range(len(self._val_test_dl))), 100, False)
#        self.train_batch_sampler = t.utils.data.BatchSampler(t.utils.data.RandomSampler(range(300)), self.num_epochs, False)
#        self.test_batch_sampler = t.utils.data.BatchSampler(t.utils.data.RandomSampler(range(100)), self.num_epochs, False)

            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._optim.zero_grad()
        forward_otpt = self._model.forward(x)
        
        loss = self._crit(forward_otpt, y)
        
        loss.backward()
        self._optim.step()
        return loss
        
        
        
    
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        
        self._optim.zero_grad()
        forward_otpt = self._model.forward(x)
        
        loss = self._crit(forward_otpt, y)
        
        loss.backward()
        self._optim.step()
        return loss, forward_otpt
        
        
        
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        losses = []
        for indices in self.train_batch_sampler:
            for idx in indices:
                img, lbl = self._train_dl[idx]
                if self._cuda:
                    img = img.cuda()
                    lbl = lbl.cuda()
                l = self.train_step(img, lbl)
                losses.append(l)
                
        return np.mean(np.array(losses))
        
        
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval()
        predictions = []
        labels = []
        losses = []
        with t.no_grad():
            for indices in self.test_batch_sampler:
                for idx in indices:
                    img, lbl = self._val_test_dl[idx]
                    if self._cuda:
                        img = img.cuda()
                        lbl = lbl.cuda()            
                    p = self._model.forward(img)
                    loss = self._crit(p, lbl)
                    losses.append(loss)
                    predictions.append(p)
                    labels.append(lbl)
                    
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        print("F1 Score - ", f1_score(labels, predictions))
        return np.mean(np.array(loss))
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        epoch_counter = 0
        train_loss = []
        val_loss = []
        
        while True:
      
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
        #TODO
        
            if epoch_counter > epochs:
                break
            
            t_loss = self.train_epoch()
            v_loss = self.val_test()
            
            train_loss.append(t_loss)
            v_loss.append(val_loss)
            
            if epoch_counter%10 == 0:
                self.save_checkpoint(epoch_counter)
                
            if val_loss[(epoch_counter - 10)%self.num_epochs] - val_loss[epoch_counter] < self._early_stopping_patience:
                break
            
        return train_loss, val_loss
            
            
            
            
                    
        
        
        
