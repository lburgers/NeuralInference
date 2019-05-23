import os
import string
import numpy as np

# make sure this is consistent with build_training_set.py
rules = ['OR', 'AND', 'THEN', "OBJ"]
total_objects = 15
objects = list(string.ascii_uppercase)[:total_objects] # first total_objects uppercase letters 
grid_size = 100
grid_row_size = 10

training_data_path = 'training_data/'
files = os.listdir(training_data_path)


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
		return grid

for file in files:

	file_path = training_data_path + file
	training_data = np.loadtxt(file_path, delimiter=',')

	for row in training_data:
		agent_history = AgentInstance()
		agent_history.import_row(row)
		grid = agent_history.build_map_data()
		trajectory = agent_history.build_trajectory()



















