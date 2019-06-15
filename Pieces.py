from func import opposite_color, color2int, move_loc, loc2int, is_pawn_starting_row, get_piece_values


class Piece(object):

	def __init__(self, name, color, square, double_advance=None, moved=None, display=None):
		self.name = name
		self.color = color
		self.square = square
		self.double_advance = double_advance
		self.moved = moved
		self.display = display
		self.value = get_piece_values(name)

	def get_direction_moves(self, board, directions, one_move=False):
		moves = []
		for direction in directions:
			loc = move_loc(self.square.loc, direction)
			while (loc in board.squares and 
				   not board.squares[loc].has_piece()):
				moves.append(loc)
				if one_move:
					break
				loc = move_loc(loc, direction)
			if (loc in board.squares and    # capturing case
				board.squares[loc].has_piece(opposite_color(self.color))):
				moves.append(loc)
		return moves

	def get_moves(self, board):
		pass

	def move(self, board, square):
		self.square = square

	def draw(self, canvas, ss, coords):
		colorint = color2int(self.color)
		colorint2 = self.color == 'white'
		column,row = self.square.loc
		column,row = loc2int(column,row)
		coords = [(colorint2-c)*colorint for c in coords]
		coords = [ss*row + ss*c if i % 2 else ss*column+ ss*c for i, c in enumerate(coords)]
		self.display = canvas.create_polygon(coords,fill=self.color,
											 outline=opposite_color(self.color),
											 tag='piece')


class Pawn(Piece):

	def __init__(self, color, square, double_advance=0, moved=None):
		super().__init__('pawn', color, square, double_advance, moved)

	def get_moves(self, board):
		moves = []
		colorint = color2int(self.color)
		column,row = self.square.loc
		for columnint in [-1, 1]:
			takeloc = (chr(ord(column)+columnint), row+colorint)
			if (takeloc in board.squares and
				board.squares[takeloc].has_piece(opposite_color(self.color))):
				moves.append(takeloc)
			enpassantloc = (chr(ord(column)+columnint),row)
			if (enpassantloc in board.squares and
				board.squares[enpassantloc].has_piece(opposite_color(self.color)) and
				board.squares[enpassantloc].get_piece().name == 'pawn' and
				board.squares[enpassantloc].get_piece().double_advance == board.move-1):
				moves.append(takeloc)
		advanceloc = (column,row+colorint)
		if advanceloc in board.squares and not board.squares[advanceloc].has_piece():
			moves.append(advanceloc)
		if is_pawn_starting_row(self.color, self.square.loc):
			advanceloc2 = (column, row+colorint*2)
			if not (board.squares[advanceloc2].has_piece() or
					board.squares[advanceloc].has_piece()):
				moves.append(advanceloc2)			
		return moves

	def move(self, board, square):
		column, row = square.loc
		my_column, my_row = self.square.loc
		if abs(row - my_row) == 2:
			self.double_advance = board.move  # for en passant
		if column != my_column and not square.has_piece():  # take and no piece there
			board.takePiece(board.squares[(column, my_row)].piece)
		super().move(board, square)

	def draw(self, canvas, ss, coords=None):
		if coords is None:
			coords = ([0.3, 0.2, 0.7, 0.2, 0.6, 0.5, 0.65, 0.52, 0.67, 0.57, 0.67, 0.62, 0.65, 0.64] +
					  [0.6, 0.65, 0.5, 0.67, 0.4, 0.65, 0.35, 0.64, 0.33, 0.62, 0.33, 0.57, 0.35, 0.52] +
					  [0.4, 0.5])
		super().draw(canvas, ss, coords)


class Rook(Piece):

	def __init__(self, color, square, double_advance=None, moved=False):
		super().__init__('rook', color, square, double_advance, moved)

	def get_moves(self, board):
		return super().get_direction_moves(board, ['up', 'down', 'left', 'right'])

	def move(self, board, square):
		self.moved = True
		super().move(board,square)

	def draw(self, canvas, ss, coords=None):
		if coords is None:
			coords = ([0.25, 0.2, 0.75, 0.2, 0.75, 0.7, 0.65, 0.7, 0.65, 0.6, 0.55, 0.6] +
					  [0.55, 0.7, 0.45, 0.7, 0.45, 0.6, 0.35, 0.6, 0.35, 0.7, 0.25, 0.7])
		super().draw(canvas, ss, coords)

class Knight(Piece):

	def __init__(self, color, square, double_advance=None, moved=None):
		super().__init__('knight', color, square, double_advance, moved)

	def get_moves(self, board):
		moves = []
		for first_direction in ['up', 'down', 'left', 'right']:
			if first_direction in ['up', 'down']:
				second_directions = ['left', 'right']
			elif first_direction in ['left', 'right']:
				second_directions = ['up', 'down']
			else:
				raise ValueError('Unrecognized direction')
			for second_direction in second_directions:
				loc = move_loc(self.square.loc, first_direction)
				loc = move_loc(loc, second_direction)
				loc = move_loc(loc, second_direction)
				if (loc in board.squares and 
					(not board.squares[loc].has_piece() or
					 board.squares[loc].has_piece(opposite_color(self.color)))):
					moves.append(loc)
		return moves

	def draw(self, canvas, ss, coords=None):
		if coords is None:
			coords = ([0.25, 0.2, 0.65, 0.2, 0.55, 0.5, 0.65, 0.6, 0.75, 0.5, 0.75, 0.6] +
					  [0.65, 0.8, 0.55, 0.75, 0.45, 0.7, 0.35, 0.65, 0.25, 0.6])
		super().draw(canvas, ss, coords)


class Bishop(Piece):

	def __init__(self, color, square, double_advance=None, moved=None):
		super().__init__('bishop', color, square, double_advance=double_advance, moved=moved)

	def get_moves(self, board):
		return super().get_direction_moves(board, ['up-left', 'up-right',
												   'down-left', 'down-right'])

	def draw(self, canvas, ss, coords=None):
		if coords is None:
			coords = ([0.35, 0.2, 0.65, 0.2, 0.55, 0.4, 0.7, 0.6, 0.5, 0.8] +
					  [0.3, 0.6, 0.45, 0.4])
		return super().draw(canvas, ss, coords)


class Queen(Piece):

	def __init__(self, color, square, double_advance=None, moved=None):
		super().__init__('queen', color, square, double_advance, moved)

	def get_moves(self,board):
		return super().get_direction_moves(board, ['up', 'down', 'left', 'right',
												   'up-left', 'up-right', 'down-left',
												   'down-right'])

	def draw(self, canvas, ss, coords=None):
		if coords is None:
			coords = ([0.25, 0.1, 0.75, 0.1, 0.65, 0.4, 0.7, 0.6, 0.7, 0.8, 0.6, 0.8] +
					  [0.6, 0.9, 0.4, 0.9, 0.4, 0.8, 0.3, 0.8, 0.3, 0.6, 0.35, 0.4])
		super().draw(canvas, ss, coords)


class King(Piece):

	def __init__(self, color, square, double_advance=None, moved=False):
		super().__init__('king', color, square, double_advance, moved)

	def get_moves(self, board):
		moves = super().get_direction_moves(board,['up', 'down', 'left', 'right'] +
												  ['up-left', 'up-right', 'down-left', 'down-right'],
												  one_move=True)
		row = 1 if self.color == 'white' else 8
		k_column, k_row = self.square.loc
		for direction,column in zip([-2, 2],['a', 'h']):
			loc = (column, row)
			if (not self.moved and
				board.squares[loc].has_piece(self.color) and
				board.squares[loc].get_piece().name == 'rook' and
				not board.squares[loc].get_piece().moved and
				not any([board.squares[(chr(i),row)].has_piece()
					     for i in range(ord(column)+1, ord(k_column))])):
				moves.append((chr(ord(k_column)+direction), row))
		return moves

	def move(self, board, square):
		self.moved = True
		k_column, k_row = self.square.loc
		column, row = square.loc
		if abs(ord(k_column)-ord(column)) == 2:  # castle
			direction = 1 if ord(column) < ord(k_column) else - 1
			rook_loc = ('a', row) if direction > 0 else ('h', row)
			rook = board.squares[rook_loc].get_piece()
			board.move_piece(rook,board.squares[(chr(ord(column)+direction), row)].loc)
		super().move(board, square)

	def draw(self, canvas, ss, coords=None):
		if coords is None:
			coords = ([0.25, 0.05, 0.75, 0.05, 0.65, 0.35, 0.7, 0.55, 0.7, 0.75, 0.55, 0.75] +
					  [0.55, 0.85, 0.6, 0.85, 0.6, 0.9, 0.55, 0.9, 0.55, 0.95, 0.45, 0.95] +
					  [0.45, 0.9, 0.4, 0.9, 0.4, 0.85, 0.45, 0.85, 0.45, 0.75] +
					  [0.4, 0.75, 0.3, 0.75, 0.3, 0.55, 0.35, 0.35])
		super().draw(canvas, ss, coords)