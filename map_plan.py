import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
This function loads and creates a 3D map tensor.
In the first plane, 0 represents an empty space, and 1 an obstacle.
In the second plane, 1 represents the goal position.
In the third plane, 1 represents the current position.

If the file name is not passed, create a sample default 20x20 map.
"""
def load_map(file_name=None):
    if file_name is None:
        # Create a sample 20x20 map
        map = np.zeros((20, 20, 3), dtype=np.float32)
        map[0, 0, 2] = 1 # Set the current position
        map[19, 10, 1] = 1 # Set the goal position
        # introduce random but some connected obstracles avoiding the goal and current position
        map[1:6, 5, 0] = 1
        map[5, 1:6, 0] = 1
        map[5:11, 10, 0] = 1
        map[10, 5:11, 0] = 1
        map[10:16, 15, 0] = 1

        return torch.from_numpy(map)
    else:
        # Load the map from the file
        map = np.load(file_name)
        return torch.from_numpy(map)
    
"""
Given a position (x,y) and the map size, this tensor returns a soft position mask.
"""
def get_position_mask(x, y, map_x, map_y, beta=1.):
    # create a list of all possible positions (x,y)
    positions = torch.zeros(map_x, map_y, 2)
    for i in range(map_x):
        for j in range(map_y):
            positions[i, j, 0] = i
            positions[i, j, 1] = j
    # flatten the positions tensor
    positions = positions.view(-1, 2)
    # create a tensor with the current position
    current = torch.tensor([x, y], dtype=torch.float32)
    # calculate the distance between the current position and all the possible positions
    distances = torch.norm(positions - current, dim=1)
    # calculate the mask
    mask = torch.exp(-beta * distances)
    # reshape the mask to the original map size
    mask = mask.view(map_x, map_y)

    return mask


"""
Given the current map and delta (dx, dy), this function returns the new position.
"""
def move_position(map, x, y, dx, dy):
    # check if there's any obstacle on the line 
    # between the current and new positions by considering both x and y simultaneously
    x_, y_ = x, y
    dx_, dy_ = dx, dy
    while True:
        x_ = x_ + np.sign(dx_)
        y_ = y_ + np.sign(dy_)

        if x_ >= map.shape[0] or y_ >= map.shape[1] or x_ < 0 or y_ < 0:
            x_ = max(0, min(x_, map.shape[0]-1))
            y_ = max(0, min(y_, map.shape[1]-1))
            break

        if map[x_, y_, 0] == 1:
            x_ = x_ - np.sign(dx_)
            y_ = y_ - np.sign(dy_)
            break

        dx_ = dx_ - np.sign(dx_)
        dy_ = dy_ - np.sign(dy_)

        if dx_ == 0 and dy_ == 0:
            break

    return x_, y_

    


