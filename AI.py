import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, pickle
from tqdm import tqdm
from Board import Board
from func import oppositeColor, loc2int, int2loc, int2color, isLastRank

def logistic(x):
	return (2. / (1 + np.exp(-x))) - 1


class AI:

	def __init__(self, color, name='rocknet', show=True):
		self.color = color
		self.network = load_network(name)
		self.network.show = show

	def make_decision(self, board):
		self.network.make_decision(board, self.color)

	def get_promotion(self, board, loc):
		return self.network.get_promotion(board, loc)


class ConnectionWeight:

	def __init__(self, weight):
		self.weight = weight


class Node:

	def __init__(self, loc):
		self.loc = loc
		self.activity = 0
		self.next_nodes = dict()
		self.previous_nodes = dict()


	def connect(self, node, weight):
		cw = ConnectionWeight(weight)
		self.next_nodes[node.loc] = cw
		node.previous_nodes[self.loc] = cw


class Network:

	piece_dict = {'pawn': 0, 'rook': 1, 'knight': 2, 'bishop': 3,
				  'queen': 4, 'king': 5}
	board_dim = 8

	def __init__(self, name='rocknet', seed=12, delta=0.1,
				 hidden_dims=[(16, 16)], show=True):
		np.random.seed(seed)
		self.name = name
		self.delta = delta
		self.show = show
		input_dim = (len(self.piece_dict)*2, self.board_dim, self.board_dim)
		output_dim = (len(self.piece_dict), self.board_dim, self.board_dim)
		self.input_layer = self.make_layer(input_dim)
		print('input')
		self.hidden_layers = []
		if hidden_dims:
			hidden_layer = self.make_layer(hidden_dims[0])
			self.connect_layers(self.input_layer, hidden_layer)
			self.hidden_layers.append(hidden_layer)
			print('hidden')
			for i, hidden_dim in enumerate(hidden_dims[1:]):
				hidden_layer = self.make_layer(hidden_dim)
				if i < len(hidden_dims) - 1:
					self.connect_layers(self.hidden_layers[-1], hidden_layer)
				self.hidden_layers.append(hidden_layer)
				print('hidden')
			self.output_layer = self.make_layer(output_dim)
			print('output')
			self.connect_layers(hidden_layer, self.output_layer)
		else:
			self.hidden_layers = []
			self.output_layer = self.make_layer(output_dim)
			self.connect_layers(self.input_layer, self.output_layer)


	def make_layer(self, shape):
		layer = np.empty(shape=shape, dtype=object).flatten()
		for i in range(layer.size):
			layer[i] = Node(loc=i)
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
			for loc, weight in node.next_nodes.items():
				update_mat[loc] += node.activity * weight.weight
		for i, node in enumerate(next_layer.flatten()):
			node.activity = logistic(update_mat[i])


	def show_activity(self):
		input_fig, input_axes = plt.subplots(self.input_layer.shape[0])
		for section, ax in zip(self.input_layer, input_axes):
			self.plot_section(section, ax)
		if self.hidden_layers:
			hidden_fig, hidden_axes = plt.subplots(len(self.hidden_layers))
			hidden_axes = hidden_axes if isinstance(hidden_axes, np.ndarray) else np.array([hidden_axes])
			for hidden_layer, ax in zip(self.hidden_layers, hidden_axes):
				self.plot_section(hidden_layer, ax)
		output_fig, output_axes = plt.subplots(self.output_layer.shape[0])
		for section, ax in zip(self.output_layer, output_axes):
			self.plot_section(section, ax)
		plt.show()


	def plot_section(self, section, ax):
		section_shape = section.shape
		ax.axis('off')
		activity_mat = np.zeros(section_shape).flatten()
		for i, node in enumerate(section.flatten()):
			activity_mat[i] = node.activity
		ax.imshow(activity_mat.reshape(section_shape))

	def train_king_hunt(self, n_games=1000):
		for n in tqdm(range(n_games)):
			board = Board()
			while board.move is not None:
				color = int2color(board.move)
				activity_mat = pieces2activity_mat(board.pieces[color], board.pieces[oppositeColor(color)])
				self.propagate(activity_mat)
				output_activity_mat = layer2activity_mat(self.output_layer)
				piece, move = activity_mat2move(output_activity_mat, board, board.pieces[color])
				print(piece.name, piece.square.loc, move)
				board.makeMove(piece, move)
				score = board.scoreKingHunt(color)
				output_loc = piece2output_layer(piece)
				self.back_propagate(self.output_layer[output_loc], score, 0)


	def back_propagate(self, node, score, i):
		if score == 0 or i == len(self.hidden_layers) + 2:
			return
		if self.hidden_layers:
			layer = self.hidden_layers[-i] if i < len(self.hidden_layers) else self.input_layer
		else:
			layer = self.input_layer
		for loc, weight in node.previous_nodes.items():
			node.previous_nodes[loc].weight = weight.weight + logistic(score)*self.delta
			self.back_propagate(layer.flatten()[loc], score / 2, i + 1)


	def make_decision(self, board, color):
		activity_mat = pieces2activity_mat(board.pieces[color], board.pieces[oppositeColor(color)])
		self.propagate(activity_mat)
		output_activity_mat = layer2activity_mat(self.output_layer)
		piece, move = activity_mat2move(output_activity_mat, board, board.pieces[color])
		board.makeMove(piece, move)


	def get_promotion(self, board, loc):
		output_activity_mat = layer2activity_mat(self.output_layer)
		return activity_mat2promotion(output_activity_mat, loc)


	def check_promotion_or_game_end(self, board):
		piece, loc = board.moves[-1]
		if piece.name == 'pawn' and isLastRank(int2color(board.move - 1), loc):
			name = self.getAIPromotion(board, loc)
			board.takePiece(piece)
			board.makePiece(name, piece.color, loc)
		outcome = board.checkCheckMate()
		if outcome:
			print(outcome)
			board.move = None


def pieces2activity_mat(my_pieces, other_pieces):
	activity_mat = np.zeros((len(Network.piece_dict)*2, Network.board_dim, Network.board_dim))
	for i, pieces in enumerate([my_pieces, other_pieces]):
		for name in pieces:
			for piece in pieces[name]:
				column, row = piece.square.loc
				column, row = loc2int(column, row)
				activity_mat[Network.piece_dict[piece.name] + i*len(Network.piece_dict), column, row] = 1
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


def piece2output_layer(piece):
	column, row = piece.square.loc
	column, row = loc2int(column, row)
	return (Network.piece_dict[piece.name], column, row)


def load_network(name):
	if op.isfile(name + '.pkl'):
		with open(name + '.pkl', 'rb') as f:
			network = pickle.load(f)
	else:
		raise ValueError('%s network does not exist' % name)
	return network


if __name__ == '__main__':
	from AI import Network
	import numpy as np
	self = Network(show=True)
	self.train_king_hunt(n_games=5)
	

