import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from Board import Board
from func import opposite_color, loc2int, int2color, is_last_rank

LETTERS = [chr(i) for i in range(65, 65+26)]
BOARD_DIM = 8
N_PIECES = 6
MAX_MOVES = 100


def logistic(x):
	return (2. / (1 + np.exp(-x))) - 1


class AI:

	def __init__(self, color, name='rock', show=False):
		self.color = color
		self.name = name
		if not op.isfile(op.join('networks', name + 'net.pkl')):
			self.train_random()
		self.network = load_network(name)
		self.network.show = show

	def make_decision(self, board):
		self.network.make_decision(board, self.color)

	def get_promotion(self, board, loc):
		return self.network.get_promotion(board, loc)

	def train_random(self, exp_n=2):
		genomes = [Genome(name=''.join([LETTERS[i] for i, b in enumerate(format(1023, '026b')) if b == '1']),
						  seed=(4334 * i)) for i in tqdm(range(2**exp_n))]
		while len(genomes) > 1:
			genomes = self.genome_tournament(genomes)
		genomes[0].network.name = self.name
		genomes[0].name = self.name
		genomes[0].network.save()
		genomes[0].save()

	def train_offspring(self, exp_n=8):
		pass

	def genome_tournament(self, genomes):
		if len(genomes) % 2:
			raise ValueError('Must have an even number of genomes for tournament')
		keep_indices = []
		for i in tqdm(range(int(len(genomes)/2))):
			board = Board()
			order = [i, len(genomes) - i - 1]
			np.random.shuffle(order)
			players = {'white': genomes[order[0]], 'black': genomes[order[1]]}
			while not board.game_over and board.move < MAX_MOVES:
				color = int2color(board.move)
				players[color].network.make_decision(board, color)
			outcome = board.check_check_mate()
			position_difference = board.score_position('white') - board.score_position('black')
			print(outcome, position_difference)
			if (outcome and 'Draw' in outcome) or position_difference == 0:
				keep_indices.append(np.random.choice([i, -i]))
			elif outcome == 'Check mate white' or position_difference > 0:
				keep_indices.append(order[0])
			elif outcome == 'Check mate black' or position_difference < 0:
				keep_indices.append(order[1])
			else:
				raise ValueError('Unrecognized outcome %s' % outcome)
		return [genomes[i] for i in keep_indices]


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
	promotion_pieces = ['rook', 'knight', 'bishop', 'queen']

	def __init__(self, layer_dims, tms, name='rock', seed=12, delta=0.1,
				 show=True):
		np.random.seed(seed)
		self.name = name
		self.delta = delta  # for backpropagation (depreciated)
		self.show = show
		self.input_layer = self.make_layer(layer_dims[0])
		self.hidden_layers = []
		if len(layer_dims) > 2:
			hidden_layer = self.make_layer(layer_dims[1])
			self.connect_layers(self.input_layer, hidden_layer, tms[0])
			self.hidden_layers.append(hidden_layer)
			for i, (hidden_dim, tm) in enumerate(zip(layer_dims[2:-1], tms[1:-1])):
				hidden_layer = self.make_layer(hidden_dim)
				if i < len(layer_dims) - 1:
					self.connect_layers(self.hidden_layers[-1], hidden_layer, tm)
				self.hidden_layers.append(hidden_layer)
			self.output_layer = self.make_layer(layer_dims[-1])
			self.connect_layers(hidden_layer, self.output_layer, tms[-1])
		else:
			self.hidden_layers = []
			self.output_layer = self.make_layer(layer_dims[-1])
			self.connect_layers(self.input_layer, self.output_layer, tms[-1])

	def make_layer(self, shape):
		layer = np.empty(shape=shape, dtype=object).flatten()
		for i in range(layer.size):
			layer[i] = Node(loc=i)
		return layer.reshape(shape)

	def connect_layers(self, layer, next_layer, tm):
		tm = tm.flatten()
		for i, node in enumerate(layer.flatten()):
			for j, next_node in enumerate(next_layer.flatten()):
				node.connect(next_node, tm[i * j])

	def save(self):
		with open(op.join('networks', self.name + 'net.pkl'), 'wb') as f:
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
			while not board.game_over and board.move < MAX_MOVES:
				color = int2color(board.move)
				activity_mat = self.pieces2activity_mat(board.pieces[color], board.pieces[opposite_color(color)])
				self.propagate(activity_mat)
				output_activity_mat = self.layer2activity_mat(self.output_layer)
				piece, move = self.activity_mat2move(output_activity_mat, board, board.pieces[color])
				print(piece.name, piece.square.loc, move)
				board.make_move(piece, move)
				score = board.score_position(color)
				output_loc = self.piece2output_layer(piece)
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
		activity_mat = self.pieces2activity_mat(board.pieces[color], board.pieces[opposite_color(color)])
		self.propagate(activity_mat)
		output_activity_mat = self.layer2activity_mat(self.output_layer)
		piece, move = self.activity_mat2move(output_activity_mat, board, board.pieces[color])
		board.make_move(piece, move)
		self.check_promotion_or_game_end(board)

	def get_promotion(self, board, loc):
		output_activity_mat = self.layer2activity_mat(self.output_layer)
		return self.activity_mat2promotion(output_activity_mat, loc)

	def check_promotion_or_game_end(self, board):
		piece, loc = board.moves[-1]
		if piece.name == 'pawn' and is_last_rank(int2color(board.move - 1), loc):
			name = self.get_promotion(board, loc)
			board.take_piece(piece)
			board.make_piece(name, piece.color, loc)
		board.check_check_mate()

	def pieces2activity_mat(self, my_pieces, other_pieces):
		activity_mat = np.zeros(self.input_layer.shape)
		for i, pieces in enumerate([my_pieces, other_pieces]):
			for name in pieces:
				for piece in pieces[name]:
					column, row = piece.square.loc
					column, row = loc2int(column, row)
					activity_mat[i, self.piece_dict[name], column, row] = 1  # output_layer.shape == n_pieces
		return activity_mat

	def layer2activity_mat(self, layer):
		activity_mat = np.zeros(layer.shape).flatten()
		for i, node in enumerate(layer.flatten()):
			activity_mat[i] = node.activity
		return activity_mat.reshape(layer.shape)

	def activity_mat2move(self, activity_mat, board, pieces):
		best_move, best_score = None, -1
		for name in pieces:
			for piece in pieces[name]:
				column, row = piece.square.loc
				start_column, start_row = loc2int(column, row)
				for move in board.get_moves(piece):
					column, row = move
					end_column, end_row = loc2int(column, row)
					score = activity_mat[self.piece_dict[piece.name], start_column, start_row, end_column, end_row]
					if score > best_score:
						best_score = score
						best_move = (piece, move)
		return best_move

	def activity_mat2promotion(self, activity_mat, loc):
		column, row = loc
		column, row = loc2int(column, row)
		return self.promotion_pieces[int(np.argmax(activity_mat[1:4, column, row, column, row]))]

	def piece2output_layer(self, piece):
		column, row = piece.square.loc
		column, row = loc2int(column, row)
		return self.piece_dict[piece.name], column, row


def load_network(name):
	if op.isfile(op.join('networks', name + 'net.pkl')):
		with open(op.join('networks', name + 'net.pkl'), 'rb') as f:
			network = pickle.load(f)
	else:
		raise ValueError('%s network does not exist' % name)
	return network


class Genome:

	DEPTH = 8
	LENGTH = int(1e6)
	MAX_LAYERS = 10
	MAX_DIMS = 5

	def __init__(self, name='rock', seed=12):
		'''
		name: String
			for versioning
		genome: String 'random' or 'load'
			'random' generates a new genome, 'load' loads previously trained genome
		seed: int
			seed for numpy random number generator
		'''
		np.random.seed(seed)
		self.name = name
		self.i = 0
		if op.isfile(op.join('genomes', self.name + 'gen.txt')):
			self.load()
		else:
			self.genome = ''.join([format(np.random.randint(2**self.DEPTH), '0%ib' % self.DEPTH)
								   for _ in range(self.LENGTH)])
		if op.isfile(op.join('networks', self.name + 'net.pkl')):
			self.load_network()
		else:
			self.make_network()

	def save(self):
		with open(op.join('genomes', self.name + 'gen.txt'), 'w') as f:
			f.write(self.genome)

	def load(self):
		with open(op.join('genomes', self.name + 'gen.txt'), 'r') as f:
			self.genome = f.readline()

	def delete(self):
		os.remove(op.join('genomes', self.name + 'gen.txt'))

	def make_network(self):
		n_layers = max([(self.next_int() % self.MAX_LAYERS) + 1, 3])
		tms = []  # transition matrices
		input_dim = (2, N_PIECES, BOARD_DIM, BOARD_DIM)
		layer_dims = [input_dim]
		for n in range(n_layers - 2):
			layer_dims.append(self.new_layer())
			tms.append(self.generate_tm(layer_dims[-2], layer_dims[-1]))
		output_dim = (N_PIECES, BOARD_DIM, BOARD_DIM, BOARD_DIM, BOARD_DIM)
		layer_dims.append(output_dim)
		tms.append(self.generate_tm(layer_dims[-2], layer_dims[-1]))
		self.network = Network(layer_dims, tms, name=self.name, show=False)

	def load_network(self):
		self.network = load_network(self.name)
	
	def next_int(self):
		self.i += self.DEPTH
		if self.i >= len(self.genome):
			raise ValueError('Genome length exceeded')
		return int(self.genome[self.i-self.DEPTH: self.i], base=2)

	def generate_tm(self, dim0, dim1):
		tm = np.zeros(dim0 + dim1).flatten()
		for i in range(tm.size):
			tm[i] = (self.next_int() - 2**(self.DEPTH - 1)) / 2**(self.DEPTH - 1)
		return tm.reshape(dim0 + dim1)

	def new_layer(self):
		n_dims = (self.next_int() % self.MAX_DIMS) + 1
		layer_dim = tuple((self.next_int() % BOARD_DIM) + 1 for _ in range(n_dims))
		return layer_dim


if __name__ == '__main__':
	ai = AI('white')
	

