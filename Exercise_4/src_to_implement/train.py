import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
df = pd.read_csv("data.csv", sep = ";")
train, test = train_test_split(df, test_size=0.2, random_state = 42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO

train_data = ChallengeDataset(train)

train_loader = t.utils.data.DataLoader(dataset=train_data, batch_size=32,
                                           drop_last=False, shuffle=True, num_workers=4)

test_data = ChallengeDataset(test)

test_loader = t.utils.data.DataLoader(dataset=train_data, batch_size=32,
                                           drop_last=False, shuffle=True, num_workers=4)


# create an instance of our ResNet model
# TODO

model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)

loss_fun = t.nn.BCELoss()

# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters())


# create an object of type Trainer and set its early stopping criterion
# TODO

model_trainer = Trainer(model,loss_fun,optim=optimizer,train_dl=train_loader,
                 val_test_dl=test_loader,
                 cuda=True, early_stopping_patience=-1)

# go, go, go... call fit on trainer
#res = #TODO


res = model_trainer.fit(epochs = 100)



# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')