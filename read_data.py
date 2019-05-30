import os
import string
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# figure out how many epochs to run w/o over fitting - use some sort of visualization
# getting a sense of the error that its making
	# research which error function to use (cross entropy)

# make sure this is consistent with build_training_set.py
rules = ['OR', 'AND', 'THEN', "OBJ"]
total_objects = 15
objects = list(string.ascii_uppercase)[:total_objects] # first total_objects uppercase letters 
grid_size = 100
grid_row_size = 10

# data parsing and preprocessing
class AgentInstance:
	def __init__(self):
		self.object_locations = {}
		self.starting_point = None
		self.rule_bias = []
		self.obj_bias = []
		self.actions = []
		self.states = []

	def index_to_location(self, index):
		row = int(index // grid_row_size)
		column = int(index % grid_row_size)
		return (row, column)

	# read row data into object
	def import_row(self, row):
		row_size = 2*total_objects + len(rules) + (2*grid_size) + 1
		actions_base_index = row_size - (2*grid_size)
		states_base_index = row_size - (grid_size)

		for i in range(total_objects):
			obj = objects[i]
			self.object_locations[obj] = row[i]
		self.starting_point = row[total_objects]
		self.rule_bias = row[total_objects+1:len(rules)+total_objects+1]
		self.obj_bias = row[len(rules)+total_objects+1: len(rules)+(2*total_objects)+1]
		self.actions = row[actions_base_index: states_base_index]
		self.states = row[states_base_index:]
	
	def build_map_data(self):
		data_block = np.zeros((total_objects+1, grid_row_size, grid_row_size), dtype=int)
		starting_xy = self.index_to_location(self.starting_point)
		data_block[0, starting_xy[0], starting_xy[1]] = 1

		for i in range(total_objects):
			obj = objects[i]
			obj_index = self.object_locations[obj]
			if obj_index == -1:
				continue

			obj_xy = self.index_to_location(obj_index)
			data_block[i+1, obj_xy[0], obj_xy[1]] = 1

		return data_block

	def build_trajectory(self):
		grid = np.zeros((grid_row_size, grid_row_size), dtype=int)
		for state in self.states:
			if state == -1:
				break
			state_xy = self.index_to_location(state)
			grid[state_xy[0], state_xy[1]] = 1
		return [grid]


# deep learning model architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.grid_conv = nn.Conv2d(total_objects+1, 5, 1)
        self.path_conv = nn.Conv2d(1, 5, 2)
       
        self.fc1 = nn.Linear(905, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, total_objects+len(rules))

    def forward(self, grids, path):
        grids = F.relu(self.grid_conv(grids))
        path = F.relu(self.path_conv(path))

        grids = grids.view(-1, self.num_flat_features(grids))
        path = path.view(-1, self.num_flat_features(path))
        x = torch.cat((grids, path), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features





training_data_path = 'training_data/'
files = [f for f in os.listdir(training_data_path) if not f.startswith('.')]

net = Net()
criterion = nn.MSELoss() # is this the best?
optimizer = optim.Adam(net.parameters(), lr=0.0001) # is this the best?

for file in files:

	print file

	file_path = training_data_path + file
	training_data = np.loadtxt(file_path, delimiter=',')
	training_in = []
	training_labels = []


	for row in training_data:
		agent_history = AgentInstance()
		agent_history.import_row(row)
		grid = agent_history.build_map_data()
		trajectory = agent_history.build_trajectory()
		
		training_in.append( (grid, trajectory) )
		label = np.concatenate((agent_history.rule_bias,agent_history.obj_bias))
		training_labels.append( label )
	
	grids = [x[0] for x in training_in]
	paths = [x[1] for x in training_in]
	grids = torch.FloatTensor(grids)
	paths = torch.FloatTensor(paths)
	training_labels = torch.FloatTensor(training_labels)


	# run model (this needs to be placed in right place w/ epochs)
	optimizer.zero_grad()
	outputs = net(grids, paths)
	loss = criterion(outputs, training_labels)
	loss.backward()
	optimizer.step()

	print loss.item()



















