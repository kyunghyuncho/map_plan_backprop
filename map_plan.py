import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


"""
Map class

1. it has three planes: (1) obstacle, (2) goal, and (3) current position
2. it supports moving to a new position given delta, taking into account the obstacles
3. it supports getting a soft position mask.
4. it supports smoothing the obstacle map.
4. it computes the gradient of the distance to the goal position
   with respect to the current position, taking into account the obstacle map.
"""
class Map:
    def __init__(self, map: torch.Tensor = None):
        super(Map, self).__init__()
        if map is None:
            # Create a sample 20x20 map
            map = torch.zeros((20, 20, 3), dtype=torch.float32)
            map[0, 0, 2] = 1 # Set the current position
            map[19, 10, 1] = 1 # Set the goal position
            # introduce random but some connected obstracles avoiding the goal and current position
            map[1:6, 5, 0] = 1
            map[10, 5:11, 0] = 1
            map[10:16, 15, 0] = 1
        self.map = map

    def return_map(self, plane, as_coordinate=False):
        if as_coordinate:
            pos = torch.where(self.map[:, :, plane] == 1)
            return pos[0], pos[1]
        return self.map[:, :, plane]

    def return_obstacle_map(self, as_coordinate=False):
        return self.return_map(0, as_coordinate=as_coordinate)
    
    def return_goal_position(self, as_coordinate=False):
        return self.return_map(1, as_coordinate=as_coordinate)
    
    def return_current_position(self, as_coordinate=False):
        return self.return_map(2, as_coordinate=as_coordinate)
    
    # get a soft position mask
    def get_position_mask(self, 
                          x: int, 
                          y: int, 
                          beta=.1):
        map_x, map_y = self.map.shape[0], self.map.shape[1]
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
        distance = torch.norm(positions - current, dim=1) ** 2
        # calculate the mask
        mask = torch.exp(-beta * distance)
        mask = mask / mask.sum()
        # reshape the mask to the original map size
        mask = mask.view(map_x, map_y)

        return mask
    
    # Gaussian smoothing of a map using a kernel of size `size` and standard deviation `sigma`.
    # the resulting image should be shifted appropriately.
    # the size of the resulting map is the same as the input map.
    # everything runs on torch directly.
    def smooth_map(self, sigma=2., size=6, plane=0):
        assert size % 2 == 0, "The size of the kernel should be even."

        # create a gaussian kernel
        kernel = torch.tensor([[np.exp(-((i - size//2)**2 + 
                                         (j - size//2)**2) / (2. * sigma**2)) 
                                         for j in range(size)] 
                                         for i in range(size)], 
                                         dtype=torch.float32)
        kernel = kernel / kernel.sum()
        # get the map
        map = self.map[:, :, plane]
        # pad the map
        map = F.pad(map, (size//2, size//2, size//2, size//2), mode='constant', value=0)
        # apply the kernel to the map using convolution
        new_map = F.conv2d(map.unsqueeze(0).unsqueeze(0), 
                           kernel.unsqueeze(0).unsqueeze(0), 
                           padding='valid')
        new_map = new_map.squeeze(0).squeeze(0)
        # remove the final column and row for consistency
        new_map = new_map[:-1, :-1]
        assert new_map.shape == self.map[:, :, plane].shape, f"The resulting map size is incorrect: {new_map.shape} != {self.map[:, :, plane].shape}"

        return new_map
    

    # move the current position to a new position
    def move_position(self, 
                      dx: int, 
                      dy: int, 
                      hypothetical=False):
        map = self.map[:, :, 0]
        # check if there's any obstacle on the line 
        # between the current and new positions by considering both x and y simultaneously
        # get the current position from the map
        x_, y_ = torch.where(self.map[:, :, 2] == 1)
        dx_, dy_ = dx, dy
        while True:
            x_ = x_ + np.sign(dx_)
            y_ = y_ + np.sign(dy_)

            if x_ >= map.shape[0] or y_ >= map.shape[1] or x_ < 0 or y_ < 0:
                x_ = max(0, min(x_, map.shape[0]-1))
                y_ = max(0, min(y_, map.shape[1]-1))
                break

            if map[x_, y_] == 1:
                x_ = x_ - np.sign(dx_)
                y_ = y_ - np.sign(dy_)
                break

            dx_ = dx_ - np.sign(dx_)
            dy_ = dy_ - np.sign(dy_)

            if dx_ == 0 and dy_ == 0:
                break
        
        if hypothetical == False:
            self.map[:, :, 2] = torch.zeros(self.map.shape[0], self.map.shape[1])
            self.map[x_, y_, 2] = 1

        return x_, y_

    # compute the distance from the current position to the goal position
    # using soft position mask and the smoothed obstacle map.
    def distance_to_goal(self, 
                         current_position_mask = None,
                         compute_grad=False, 
                         obstacle_strength=100.,
                         beta=1.,
                         sigma=1.5,
                         size=4):
        if current_position_mask is None:
            current_position_mask = self.get_position_mask(torch.where(self.map[:, :, 2] == 1)[0], 
                                                        torch.where(self.map[:, :, 2] == 1)[1],
                                                        beta=beta)
        
        current_position_mask.requires_grad = compute_grad
        current_position_mask.grad = None

        smoothed_obstacle_map = self.smooth_map(plane=0, sigma=sigma, size=size)
        smoothed_obstacle_map.requires_grad = False
        smoothed_goal_map = self.get_position_mask(torch.where(self.map[:, :, 1] == 1)[0], 
                                                   torch.where(self.map[:, :, 1] == 1)[1],
                                                   beta=beta)
        smoothed_goal_map.requires_grad = False
        distance = torch.abs(current_position_mask - smoothed_goal_map)
        distance = distance + obstacle_strength * smoothed_obstacle_map
        distance = current_position_mask.clone().detach() * distance
        distance = distance.sum()

        if compute_grad:
            distance.backward()
            return distance, current_position_mask.grad
        
        return distance


