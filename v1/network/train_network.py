import os
import string
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# add cuda processing

# figure out how many epochs to run w/o over fitting - use some sort of visualization
# getting a sense of the error that its making
	# research which error function to use (cross entropy)

# make sure this is consistent with build_training_set.py
rules = ['OR', 'AND', 'THEN', "OBJ"]
total_objects = 15
objects = list(string.ascii_uppercase)[:total_objects] # first total_objects uppercase letters 
grid_size = 100
grid_row_size = 10

batch_size = 5
test_split = 0.05
shuffle_dataset = True

SAVE_TEST_OUTPUT = True

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


# dataset for entire training data
class AgentDataset(Dataset):

	# load all csv data into dataset
	def __init__(self, root_dir):

		data = np.array([])
		files = [f for f in os.listdir(root_dir) if not f.startswith('.')]
		for file in files:
			file_path = root_dir + file
			file_data = np.loadtxt(file_path, delimiter=',')
			data = np.vstack([data, file_data]) if data.size else file_data

		self.agent_history = data
		self.root_dir = root_dir

	def __len__(self):
		return len(self.agent_history)

	# preprocess csv row into grid, path, and bias
	def __getitem__(self, idx):
		agent_history = AgentInstance()
		agent_history.import_row(self.agent_history[idx])

		grid = agent_history.build_map_data()
		path = agent_history.build_trajectory()
		bias = np.concatenate((agent_history.rule_bias, agent_history.obj_bias))
		sample = {'grid': torch.FloatTensor(grid), 'path': torch.FloatTensor(path), 'bias': torch.FloatTensor(bias)}
		
		return sample


# deep learning model architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.grid_conv = nn.Conv2d(total_objects+1, 5, 1, stride=1)
        self.path_conv = nn.Conv2d(1, 5, 4, stride=2)
       
		# add pool ?

        self.fc1 = nn.Linear(580, 120)
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

print("LOADING DATASET")
# load dataset - define batch_size, workers, shuffle
dataset = AgentDataset('../training_data/')

test_size = int(test_split * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size]) 

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=shuffle_dataset, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=shuffle_dataset, num_workers=4)

net = Net()

criterion = nn.MSELoss() # is this the best?
optimizer = optim.Adam(net.parameters(), lr=0.0001) # is this the best?

print("TRAINING DATASET")
for epoch in range(6):

	running_loss = 0.0
	input_size = grid_size*(total_objects+2)
	bias_size = total_objects+len(rules)
	test_outputs = np.zeros( ( test_size, input_size + 2*bias_size ) )

	for i, test_data in enumerate(test_dataloader):

		grids = test_data['grid']
		paths = test_data['path']
		biases = test_data['bias']

		outputs = net(grids, paths)
		loss = criterion(outputs, biases)
		running_loss += loss.item()

		if SAVE_TEST_OUTPUT:
			test_outputs[i*batch_size:(i+1)*batch_size, :grid_size] = paths.view(-1, net.num_flat_features(paths))
			test_outputs[i*batch_size:(i+1)*batch_size, grid_size:input_size] = grids.view(-1, net.num_flat_features(grids))
			test_outputs[i*batch_size:(i+1)*batch_size, input_size: input_size + bias_size] = outputs.detach().numpy()
			test_outputs[i*batch_size:(i+1)*batch_size, input_size + bias_size:] = biases

	# print test stats
	print('[%d] loss: %f' % (epoch, running_loss / test_size))
	if SAVE_TEST_OUTPUT:
		np.savetxt("test_outputs/epoch_%d.csv" % (epoch), test_outputs, delimiter=",")


	for i, train_data in enumerate(train_dataloader):

		grids = train_data['grid']
		paths = train_data['path']
		biases = train_data['bias']

		# run model 
		# zero parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize

		outputs = net(grids, paths)
		loss = criterion(outputs, biases)
		loss.backward()
		optimizer.step()


torch.save(net.state_dict(), './model.pt')

print ('finished')


















