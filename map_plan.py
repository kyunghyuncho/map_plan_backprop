import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

def create_sample_map():
    # Create a sample 20x20 map
    map = torch.zeros((20, 20, 3), dtype=torch.float32)
    map[0, 0, 2] = 1 # Set the current position
    map[19, 10, 1] = 1 # Set the goal position
    # introduce random but some connected obstracles avoiding the goal and current position
    map[1:6, 5, 0] = 1
    map[10, 5:11, 0] = 1
    map[10:16, 15, 0] = 1
    return map

def get_soft_position(x, y, size=(20,20), beta=1.):
    # list all possible coordinates in a grid of `size` by `size`
    # in order to compute the distance to each point in the grid from (x,y)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    x_grid = torch.arange(size[0], dtype=torch.float32)
    y_grid = torch.arange(size[1], dtype=torch.float32)
    X, Y = torch.meshgrid(x_grid, y_grid)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    # compute the distance to each point in the grid from (x,y)
    dist = (X - x)**2 + (Y - y)**2
    dist = torch.exp(-beta * dist)
    dist = dist / dist.sum()
    # rearrange `dist` to a grid
    dist = dist.reshape(size[0], size[1])
    return dist

def get_hard_position(soft_position, argmax=False):
    size = soft_position.shape
    x_grid = torch.arange(size[0], dtype=torch.float32)
    y_grid = torch.arange(size[1], dtype=torch.float32)
    # get the marginal soft_position weights over x and y, respectively
    x_weights = soft_position.sum(dim=1)
    y_weights = soft_position.sum(dim=0)
    # compute the weighted sums over x and y, respectively
    if argmax:
        x = x_weights.argmax()
        y = y_weights.argmax()
    else:
        x = (x_grid * x_weights).sum()
        y = (y_grid * y_weights).sum()

    return x, y

def get_collision_score(obstacle_map, soft_position):
    # compute the weighted sum of the obstacle map using the soft_position
    collision_score = (obstacle_map * soft_position).sum()
    return collision_score

def move_soft_position(soft_position, dx, dy, size=(20,20), beta=1.):
    # get the hard coordinate
    x, y = get_hard_position(soft_position)
    # get the new hard coordinate
    x += dx
    y += dy    
    # get the new soft position
    new_soft_position = get_soft_position(x, y, size=size, beta=beta)

    return new_soft_position

def plot_soft_position(soft_position: torch.Tensor, ax=None, cmap=None, alpha=1.0):
    if ax is None:
        fig, ax = plt.subplots()
    if cmap is None:
        cmap = 'hot'

    ax.imshow(soft_position, cmap=cmap, alpha=alpha, interpolation='nearest')
    # put both x and y axes and use integer ticks
    ax.set_xticks(np.arange(0, soft_position.shape[1], 1))
    ax.set_yticks(np.arange(0, soft_position.shape[0], 1))
    ax.set_xticks(np.arange(-.5, soft_position.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, soft_position.shape[0], 1), minor=True)
    # put soft grid lines 
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    return ax