import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import torch
import torchvision.utils as utils
import torch.nn.functional as F
import torch.nn as nn

def get_binary_receptive_field(net, image_size, i, j):
    inputs = torch.randn(32, 1, image_size[0], image_size[1], requires_grad=True)
    net.eval()
    net.to('cpu')
    outputs = net(inputs)
    loss = outputs[0,0,i,j]
    loss.backward()
    rfield = torch.abs(inputs.grad[0, 0]) > 0
    return rfield


def save_model(model, filename):
    try:
        do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
        if do_save == 'yes':
            torch.save(model.state_dict(), filename)
            print('Model saved to %s.' % (filename))
        else:
            print('Model not saved.')
    except:
        raise Exception('The notebook should be run or validated with skip_training=True.')


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()


def plot_images(images, ncol=12, figsize=(8,8), cmap=plt.cm.Greys, clim=[0,1]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    grid = utils.make_grid(images, nrow=ncol, padding=0, normalize=False).cpu()
    ax.imshow(grid[0], cmap=cmap, clim=clim)
    display.display(fig)
    plt.close(fig)


def plot_generated_samples(samples, ncol=12):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('off')
    ax.imshow(
        np.transpose(
            utils.make_grid(samples, nrow=ncol, padding=0, normalize=True).cpu(),
            (1,2,0)
        )
     )
    display.display(fig)
    plt.close(fig)