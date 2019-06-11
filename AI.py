import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import time
import pickle
from func import oppositeColor, loc2int, int2loc, int2color, isLastRank

def logistic(x):
	return 1. / (1 + np.exp(-x))


class AI:

	def __init__(self, color, name='rocknet', show=True):
		self.color = color
		self.network = Network(name=name, show=show)

	def make_decision(self, board):
		activity_mat = pieces2activity_mat(board.pieces[self.color])
		self.network.propagate(activity_mat)
		output_activity_mat = layer2activity_mat(self.network.output_layer)
		piece, move = activity_mat2move(output_activity_mat, board, board.pieces[self.color])
		board.makeMove(piece, move)

	def get_promotion(self, board, loc):
		output_activity_mat = layer2activity_mat(self.network.output_layer)
		return activity_mat2promotion(output_activity_mat, loc)


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

	piece_dict = {'pawn': 0, 'rook': 1, 'knight': 2, 'bishop': 3,
				  'queen': 4, 'king': 5}
	board_dim = 8

	def __init__(self, name='rocknet', seed=12, show=True, 
				 hidden_dims=[2]):
		np.random.seed(seed)
		if op.isfile(name + '.pkl'):
			with open(name + '.pkl', 'rb') as f:
				self = pickle.load(f)
		else:
			self.name = name
			self.show = show
			base_dim = (len(self.piece_dict), self.board_dim, self.board_dim)
			self.input_layer = self.make_layer(base_dim)
			print('input')
			self.hidden_layers = []
			if hidden_dims:
				hidden_layer = self.make_layer(
					tuple([int(dim*hidden_dims[0]) if i > 0 else dim for i, dim in enumerate(base_dim)]))
				self.connect_layers(self.input_layer, hidden_layer)
				self.hidden_layers.append(hidden_layer)
				print('hidden')
				for i, hidden_dim in enumerate(hidden_dims[1:]):
					hidden_layer = self.make_layer(
						tuple([int(dim*hidden_dim) if i > 0 else dim for i, dim in enumerate(base_dim)]))
					if i < len(hidden_dims) - 1:
						self.connect_layers(self.hidden_layers[-1], hidden_layer)
					self.hidden_layers.append(hidden_layer)
					print('hidden')
				self.output_layer = self.make_layer(base_dim)
				print('output')
				self.connect_layers(hidden_layer, self.output_layer)
			else:
				self.output_layer = self.make_layer(base_dim)
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
			pickle.dump(self, f)


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


def pieces2activity_mat(pieces):
	activity_mat = np.zeros((len(Network.piece_dict), Network.board_dim, Network.board_dim))
	for name in pieces:
		for piece in pieces[name]:
			column, row = piece.square.loc
			column, row = loc2int(column, row)
			activity_mat[Network.piece_dict[piece.name], column, row] = 1
	return activity_mat


def layer2activity_mat(layer):
	base_dim = (len(Network.piece_dict), Network.board_dim, Network.board_dim)
	activity_mat = np.zeros(base_dim).flatten()
	for i, node in enumerate(layer.flatten()):
		activity_mat[i] = node.activity
	return activity_mat.reshape(base_dim)


def activity_mat2move(activity_mat, board, pieces):
	best_move, best_score = None, -1
	for name in pieces:
		for piece in pieces[name]:
			column, row = piece.square.loc
			column, row = loc2int(column, row)
			stay_score = activity_mat[Network.piece_dict[piece.name], column, row]
			for move in board.getMoves(piece):
				column, row = move
				column, row = loc2int(column, row)
				move_score = activity_mat[Network.piece_dict[piece.name], column, row]
				if move_score - stay_score > best_score:
					best_score = move_score - stay_score
					best_move = (piece, move)
	return best_move


def activity_mat2promotion(activity_mat, loc):
	column, row = loc
	column, row = loc2int(column, row)
	return Network.piece_dict[np.argmax(activity_mat[:, column, row])]


if __name__ == '__main__':
	from AI import AI
	from Board import Board
	board = Board()
	ai = AI()
	ai.make_decision
	from AI import Network
	import numpy as np
	self = Network()
	self.propagate(np.random.random((6, 8, 8)))
	

