import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import time
import pickle
from func import oppositeColor, int2loc, int2color, isLastRank

def logistic(x):
	return 1. / (1 + np.exp(-x))


class AI:

	def __init__(self, color):
		self.color = color

	def make_decision(self, board):
		move, i = None, 0
		while move is None:
			piece = board.getPieces(self.color)[i]
			moves = board.getMoves(piece)
			if moves:
				move = moves[0]
		board.makeMove(piece, move)


class Node:

	def __init__(self, location):
		self.location = location
		self.activity = 0
		self.next_nodes = dict()
		self.previous_nodes = dict()


	def connect(self, node, weight):
		self.next_nodes[node.location] = weight
		node.previous_nodes[self.location] = weight


class Network:

	def __init__(self, name='rocknet', seed=12, show=True, input_dim=(5, 8, 8), output_dim=(5, 8, 8), 
				 hidden_dims=[(5, 16, 16), (5, 32, 32), (5, 64, 64), (5, 32, 32), (5, 16, 16)]):
		np.random.seed(seed)
		if op.isfile(name + '.pkl'):
			with open(name + '.pkl', 'rb') as f:
				self = pickle.load(f)
		else:
			self.name = name
			self.show = show
			self.input_layer = self.make_layer(input_dim)
			self.hidden_layers = []
			if hidden_dims:
				hidden_layer = self.make_layer(hidden_dims[0])
				self.connect_layers(self.input_layer, hidden_layer)
				self.hidden_layers.append(hidden_layer)
				for i, hidden_dim in enumerate(hidden_dims[1:]):
					hidden_layer = self.make_layer(hidden_dims[-1])
					if i < len(hidden_dims) - 1:
						self.connect_layers(self.hidden_layers[-1], hidden_layer)
					self.hidden_layers.append(hidden_layer)
				self.output_layer = self.make_layer(output_dim)
				self.connect_layers(hidden_layer, self.output_layer)
			else:
				self.output_layer = self.make_layer(output_dim)
				self.connect_layers(self.input_layer, self.output_layer)


	def make_layer(self, shape):
		layer = np.empty(shape=shape, dtype=object).flatten()
		for i in range(layer.size):
			layer[i] = Node(location=i)
		return layer.reshape(shape)


	def connect_layers(self, layer, next_layer):
		for node in layer.flatten():
			for next_node in next_layer.flatten():
				node.connect(next_node, np.random.random()*2 - 1)


	def save(self):
		with open(self.name + '.pkl', 'wb') as f:
			pickle.dumps(self, f)


	def propagate(self, input_activity):
		if input_activity.shape != self.input_layer.shape:
			raise ValueError('Input activity dimension mismatch')
		input_activity = input_activity.flatten()
		for i, node in enumerate(self.input_layer.flatten()):
			node.activity = input_activity[i]
		if self.hidden_layers:
			self.propagate_layer(self.input_layer, self.hidden_layers[0])
			for i, hidden_layer in enumerate(self.hidden_layers[1:]):
				self.propagate_layer(self.hidden_layers[i], hidden_layer)
			self.propagate_layer(self.hidden_layers[-1], self.output_layer)
		else:
			self.propagate_layer(self.input_layer, self.output_layer)
		if self.show:
			self.show_activity()


	def propagate_layer(self, layer, next_layer):
		update_mat = np.zeros(shape=next_layer.shape).flatten()
		for node in layer.flatten():
			for location, weight in node.next_nodes.items():
				update_mat[location] += node.activity * weight
		for i, node in enumerate(next_layer.flatten()):
			node.activity = logistic(update_mat[i])


	def show_activity(self):
		fig, axes = plt.subplots(self.input_layer.shape[0], len(self.hidden_layers) + 2)
		self.plot_layer(self.input_layer, axes[:, 0])
		for i, hidden_layer in enumerate(self.hidden_layers):
			self.plot_layer(hidden_layer, axes[:, i + 1])
		self.plot_layer(self.output_layer, axes[:, -1])
		plt.show()


	def plot_layer(self, layer, axes):
		for section, ax in zip(layer, axes):
			ax.axis('off')
			activity_mat = np.zeros(layer.shape[1:]).flatten()
			for i, node in enumerate(section.flatten()):
				activity_mat[i] = node.activity
			ax.imshow(activity_mat.reshape(layer.shape[1:]))


if __name__ == '__main__':
	from AI import Network
	import numpy as np
	self = Network()
	self.propagate(np.random.random((5, 8, 8)))


