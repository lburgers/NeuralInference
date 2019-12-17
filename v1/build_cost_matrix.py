#! /usr/bin/python2.7
from Bishop import *
import numpy as np

# change this depending on grid
grid_size = 100

cost_matrix = np.zeros((grid_size, grid_size))

def update_obs(start=None, end=None):
	assert start != None and end != None, "needs start and end"
	obs.SetStartingPoint(start, 0)
	obs.Plr.Map.AddExitState(end)
	obs.Plr.BuildPlanner()
	return

obs = LoadObserver("Map", 0, 0)


# calculate the cost of moving from all locations to another
for i in range(grid_size):
	for j in range(grid_size):
		print(i, j)
		if cost_matrix[i, j] == 0:
			update_obs(i, j)
			cost = obs.Plr.CostMatrix[ 0 , 2 ] # cost from start state to end state on one obj
			cost_matrix[i, j] = cost
			cost_matrix[j, i] = cost

# save this data so it can be accessed quickly when building trainging data
np.savetxt("cost_matrix.csv", cost_matrix, delimiter=",")