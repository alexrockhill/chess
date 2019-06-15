from Pieces import Pawn, Rook, Knight, Bishop, Queen, King
from func import opposite_color, int2color, loc2int, colors, name_order, row_colors
import numpy as np


class Square:

	def __init__(self, column, row, color, piece=None, rect=None, display=None):
		self.loc = (column, row)
		self.color = color
		self.piece = piece
		self.rect = rect
		self.display = display

	def has_piece(self, color=None):
		return (self.piece is not None if color is None else 
				self.piece is not None and self.piece.color == color)

	def get_piece(self):
		return self.piece

	def remove_piece(self):
		if self.piece:
			self.piece.square = None
		self.piece = None


class Board:

	name_dict = {'pawn': Pawn, 'rook': Rook, 'knight': Knight,
				 'bishop': Bishop, 'queen': Queen, 'king': King}

	score_dict = {'pawn': 1, 'rook': 5, 'knight': 3, 'bishop': 3,
				  'queen': 9, 'king': 0}

	def __init__(self, move=0, moves=None, positions=None, squares=None, pieces=None):
		self.game_over = False
		self.move = move  #white even, black odd
		self.moves = [] if moves is None else moves
		self.positions = [] if positions is None else positions
		if squares is None and pieces is None:
			self.make_squares()
			self.init_pieces()
		else:
			self.squares = squares
			self.pieces = pieces

	def init_pieces(self):
		self.pieces = {color:{name:[] for name in self.name_dict} for color in colors}
		for name,column in zip(name_order,[chr(i) for i in range(97,105)]):
			for color in row_colors:
				self.make_piece('pawn',color,(column, row_colors[color]['pawn']))
				self.make_piece(name,color,(column, row_colors[color]['piece']))
		self.positions.append(self.get_position())

	def make_piece(self, name, color, loc):
		piece = self.name_dict[name](color, self.squares[loc])
		self.pieces[color][name].append(piece)
		self.squares[loc].piece = piece

	def make_squares(self):
		self.squares = {}
		for row in range(1,9):
			for column in [chr(i) for i in range(97, 105)]:
				color = ('black' if (row % 2 and ord(column) % 2) or
									 not (row % 2 or ord(column) % 2) else 'white')
				self.squares[(column, row)] = Square(column, row, color)

	def make_move(self,piece, loc):
		self.move += 1
		self.moves.append((piece, loc))
		self.positions.append(self.get_position())
		self.move_piece(piece, loc)

	def move_piece(self,piece,loc):
		if self.squares[loc].piece:
			self.take_piece(self.squares[loc].piece)
		self.squares[piece.square.loc].piece = None
		self.squares[loc].piece = piece
		piece.move(self,self.squares[loc])

	def get_pieces(self, color=None):
		return (self._get_pieces(color) if color else
				[piece for color in colors for piece in self._get_pieces(color)])

	def _get_pieces(self, color):
		pieces = []
		for name in self.pieces[color]:
			pieces += self.pieces[color][name]
		return pieces

	def take_piece(self, piece):
		self.pieces[piece.color][piece.name].remove(piece)

	def get_king_loc(self, color):
		return self.pieces[color]['king'][0].square.loc

	def get_moves(self, piece):
		return self.check_check(piece, piece.get_moves(self))

	def get_position(self):
		position = {}
		for piece in self.get_pieces():
			position[piece] = piece.square.loc
		return position

	def set_position(self, position):
		for loc in self.squares:
			self.squares[loc].piece = None
		for piece in position:
			self.squares[position[piece]].piece = piece
			piece.square = self.squares[position[piece]]
			if not piece in self.pieces[piece.color][piece.name]: # if it was taken in simulation
				self.pieces[piece.color][piece.name].append(piece)

	def check_check(self, piece, moves):
		position = self.get_position()
		for loc in moves.copy():
			self.move_piece(piece, loc)
			if self.in_check(piece.color):
				moves.remove(loc)
			self.set_position(position)
		return moves

	def check_check_mate(self):
		if self.check_repeated_moves():
			self.game_over = True
			return 'Draw by repetition'
		color = int2color(self.move) # potentially checkmated player's turn
		for piece in self.get_pieces(color):
			if len(self.get_moves(piece)) > 0:
				return False
		self.game_over = True
		return 'Check mate %s' % opposite_color(color) if self.in_check(color) else 'Draw by stalemate'

	def check_repeated_moves(self):
		if len(self.positions) > 8:
			if (self.check_same_position(-1, -5) and self.check_same_position(-2, -6) and
				self.check_same_position(-3, -7) and self.check_same_position(-4, -8)):
				return True
		return False

	def check_same_position(self, ind0, ind1):
		for piece in self.positions[ind0]:
			if (not piece in self.positions[ind1] or
					self.positions[ind0][piece] != self.positions[ind1][piece]):
				return False
		return True

	def in_check(self,color):
		other_color = opposite_color(color)
		for piece in self.get_pieces(other_color):
			if (piece.name != 'king' and  # king can't give check
				self.get_king_loc(color) in piece.get_moves(self)):
				return True
		return False

	def draw(self,canvas,ss):
		canvas.delete('all')
		for loc in self.squares:
			square = self.squares[loc]
			column, row = loc
			column, row = loc2int(column, row)
			square.rect = canvas.create_rectangle(ss*column, ss*row, ss*column+ss, ss*row+ss,
												  outline='black', fill=square.color,
												  tag='board')
		for piece in self.get_pieces():
			piece.draw(canvas, ss)
		canvas.lower('board')

	def score_position(self, color):
		score = 0
		opposite_king_loc = self.get_king_loc(opposite_color(color))
		opposite_king_moves = self.get_moves(self.pieces[opposite_color(color)]['king'][0])
		for name in self.pieces[color]:
			score += len(self.pieces[color][name])*self.score_dict[name]  # Piece score
			for piece in self.pieces[color][name]:
				for move in self.get_moves(piece):
					if move == opposite_king_loc and not opposite_king_moves:  # checkmate
						score += np.inf
					elif move in opposite_king_moves or move == opposite_king_loc:  # king pressure
						score += 1
		for name in self.pieces[opposite_color(color)]:
			score -= len(self.pieces[opposite_color(color)][name])*self.score_dict[name]  # Opponent piece negative score
		return score