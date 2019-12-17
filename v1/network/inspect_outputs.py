import os
import string
import numpy as np
from collections import defaultdict


rules = ['OR', 'AND', 'THEN', "OBJ"]
total_objects = 15
objects = list(string.ascii_uppercase)[:total_objects] # first total_objects uppercase letters 
grid_size = 100
grid_row_size = 10

root_dir = './test_outputs/'

def print_bias(bias):
	if bias != None:
		labels = rules + objects
		j = -1
		for i in range(len(labels)):
			label = labels[i]
			print("\033[1;0m%8s" % (label)),

			if i % 6 == 5 or i == len(labels) - 1:
				print("\n"),
				for j in range(j+1, i+1):
					b = bias[j]
					print("%8.3f" % (b)),
				print("\n"),


def print_data(grid, path, predicted_bias=None, true_bias=None):
	full_objects = ['S'] + objects

	for x in range(grid_row_size):
		for y in range(grid_row_size):

			for i in range(total_objects+1):
				obj = full_objects[i]
				to_print = None

				if path[x,y] == 1 and grid[i, x, y] == 1:
					to_print = obj
					break
				elif grid[i, x, y] == 1:
					to_print = obj.lower()
					break
				elif path[x,y] == 1:
					to_print = "1"
				else:
					to_print = "0"

			if to_print.isupper() or to_print == "1":
				print("\033[1;31m%s " % to_print),
			elif to_print.islower():
				print("\033[1;34m%s " % to_print),
			else:
				print("\033[1;0m%s " % to_print),

		print("\n")

	if predicted_bias != None:
		print("\033[1;0m\nPredicted biases: ")
		print_bias(predicted_bias)
	if true_bias != None:
		print("\nTrue biases: ")
		print_bias(true_bias)

input_size = grid_size*(total_objects+2)
bias_size = total_objects+len(rules)

while True:

	print("Starting Inspection")

	epoch_number = input("Which Epoch : ")

	if type(epoch_number) != int:
		exit(1)

	file_path = root_dir + ("epoch_%d.csv" % epoch_number)

	print("Loading file %s" % file_path)

	file_data = np.loadtxt(file_path, delimiter=',')

	paths = np.reshape(file_data[:,:grid_size], (file_data.shape[0], grid_row_size, grid_row_size))
	grids = np.reshape(file_data[:,grid_size:input_size], (file_data.shape[0], total_objects+1, grid_row_size, grid_row_size))
	predicted_biases = file_data[:, input_size:input_size+bias_size]
	true_biases = file_data[:, input_size+bias_size:]

	# index data by map
	print("Sorting by map and bias")
	sorted_grids = defaultdict(list)
	sorted_biases = defaultdict(list)
	indices = []
	for i in range(len(grids)):
		grid = grids[i]
		bias = true_biases[i]

		grid.flags.writeable = False
		bias.flags.writeable = False

		grid_index = hash(grid.data)
		bias_index = hash(bias.data)

		if bias_index not in sorted_grids[grid_index]:
			sorted_grids[grid_index].append( bias_index )

		sorted_biases[bias_index].append( i )
		indices.append(grid_index)

		grid.flags.writeable = True
		bias.flags.writeable = True

	indices = list(set(indices))

	while True:

		map_index = input("There are %d maps, choose which map : " % len(indices))

		if type(map_index) != int:
			break
		elif map_index >= len(indices):
			print("Index out of bounds (len %d)" % len(indices))
			continue
		else:
			which_bias = input("There are %d biases for this map, choose which bias set : " % len(sorted_grids[indices[map_index]]))
			bias_index = sorted_grids[indices[map_index]][which_bias]
			which_data_point = input("There are %d data points for this bias, choose which to view : " % len(sorted_biases[bias_index]))
			index = sorted_biases[bias_index][which_data_point]
			print_data(grids[index], paths[index], predicted_biases[index], true_biases[index])

