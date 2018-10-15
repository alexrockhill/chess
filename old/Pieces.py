from func import (oppositeColor, color2int, int2color, moveloc, loc2int, int2loc,
				  makePiece, isPawnStartingRow, isLastRank, colors)

class Piece(object):

	def __init__(self,name,color,doubleAdvance=None,moved=None,display=None):
		self.name = name
		self.color = color
		self.doubleAdvance = doubleAdvance
		self.moved = moved
		self.display = display

	def getDirectionMoves(self,board,directions,oneMove=False):
		moves = []
		for direction in directions:
			loc = moveloc(board.getLoc(self),direction)
			while (loc in board.squares and 
				   not board.squares[loc].hasPiece()):
				moves.append(loc)
				if oneMove:
					break
				loc = moveloc(loc,direction)
			if (loc in board.squares and    #capturing case
				board.squares[loc].hasPiece(oppositeColor(self.color))):
				moves.append(loc)
		return moves

	def getMoves(self,board):
		return []

	def move(self,board,square):
		self.moved = True

	def draw(self,board,canvas,ss,coords):
		colorint = color2int(self.color)
		colorint2 = self.color=='white'
		column,row = board.getLoc(self)
		column,row = loc2int(column,row)
		coords = [(colorint2-c)*colorint for c in coords]
		coords = [ss*row + ss*c if i%2 else ss*column+ ss*c for i,c in enumerate(coords)]
		self.display = canvas.create_polygon(coords,fill=self.color,
											 outline=oppositeColor(self.color),
											 tag='piece')
class Pawn(Piece):

	def __init__(self,color,doubleAdvance=0,moved=None):
		super().__init__('pawn',color,doubleAdvance,moved)

	def getMoves(self,board):
		moves = []
		colorint = color2int(self.color)
		column,row = board.getLoc(self)
		for columnint in [-1,1]:
			takeloc = (chr(ord(column)+columnint),row+colorint)
			if (takeloc in board.squares and
				board.squares[takeloc].hasPiece(oppositeColor(self.color))):
				moves.append(takeloc)
			enpassantloc = (chr(ord(column)+columnint),row)
			if (enpassantloc in board.squares and
				board.squares[enpassantloc].hasPiece(oppositeColor(self.color)) and
				board.squares[enpassantloc].getPiece().name == 'pawn' and
				board.squares[enpassantloc].getPiece().doubleAdvance == board.move-1):
				moves.append(takeloc)
		advanceloc = (column,row+colorint)
		if advanceloc in board.squares and not board.squares[advanceloc].hasPiece():
			moves.append(advanceloc)
		if isPawnStartingRow(self.color,board.getLoc(self)):
			advanceloc2 = (column,row+colorint*2)
			if not board.squares[advanceloc2].hasPiece():
				moves.append(advanceloc2)			
		return moves

	def move(self,board,square):
		column,row = square.loc
		my_column,my_row = board.getLoc(self)
		if abs(row - my_row) == 2:
			self.doubleAdvance = board.move # for en passant
		if column != my_column and not square.hasPiece(): # take and no piece there
			board.takePiece(board.squares[(column,my_row)].piece)

	def draw(self,board,canvas,ss):
		coords = ([0.3,0.2,0.7,0.2,0.6,0.5,0.65,0.52,0.67,0.57,0.67,0.62,0.65,0.64] + 
				  [0.6,0.65,0.5,0.67,0.4,0.65,0.35,0.64,0.33,0.62,0.33,0.57,0.35,0.52] +
				  [0.4,0.5])
		super().draw(board,canvas,ss,coords)

class Rook(Piece):

	def __init__(self,color,doubleAdvance=None,moved=False):
		super().__init__('rook',color,doubleAdvance,moved)

	def getMoves(self,board):
		return super().getDirectionMoves(board,['up','down','left','right'])

	def draw(self,board,canvas,ss):
		coords = ([0.25,0.2,0.75,0.2,0.75,0.7,0.65,0.7,0.65,0.6,0.55,0.6] +
				  [0.55,0.7,0.45,0.7,0.45,0.6,0.35,0.6,0.35,0.7,0.25,0.7])
		super().draw(board,canvas,ss,coords)

class Knight(Piece):

	def __init__(self,color,doubleAdvance=None,moved=None):
		super().__init__('knight',color,doubleAdvance,moved)

	def getMoves(self,board):
		moves = []
		row, column = board.getLoc(self)
		for firstDirection in ['up','down','left','right']:
			if firstDirection in ['up','down']:
				secondDirections = ['left','right']
			elif firstDirection in ['left','right']:
				secondDirections = ['up','down']
			for secondDirection in secondDirections:
				loc = moveloc(board.getLoc(self),firstDirection)
				loc = moveloc(loc,secondDirection)
				loc = moveloc(loc,secondDirection)
				if (loc in board.squares and 
					(not board.squares[loc].hasPiece() or
					 board.squares[loc].hasPiece(oppositeColor(self.color)))):
					moves.append(loc)
		return moves

	def draw(self,board,canvas,ss):
		coords = ([0.25,0.2,0.65,0.2,0.55,0.5,0.65,0.6,0.75,0.5,0.75,0.6] +
				  [0.65,0.8,0.55,0.75,0.45,0.7,0.35,0.65,0.25,0.6])
		super().draw(board,canvas,ss,coords)

class Bishop(Piece):

	def __init__(self,color,doubleAdvance=None,moved=None):
		super().__init__('bishop',color,doubleAdvance=doubleAdvance,moved=moved)

	def getMoves(self,board):
		return super().getDirectionMoves(board,['up-left','up-right',
												'down-left','down-right'])

	def draw(self,board,canvas,ss):
		coords = ([0.35,0.2,0.65,0.2,0.55,0.4,0.7,0.6,0.5,0.8] +
				  [0.3,0.6,0.45,0.4])
		return super().draw(board,canvas,ss,coords)

class Queen(Piece):

	def __init__(self,color,doubleAdvance=None,moved=None):
		super().__init__('queen',color,doubleAdvance,moved)

	def getMoves(self,board):
		return super().getDirectionMoves(board,['up','down','left','right',
												'up-left','up-right','down-left',
												'down-right'])

	def draw(self,board,canvas,ss):
		coords = ([0.25,0.1,0.75,0.1,0.65,0.4,0.7,0.6,0.7,0.8,0.6,0.8] +
				  [0.6,0.9,0.4,0.9,0.4,0.8,0.3,0.8,0.3,0.6,0.35,0.4])
		super().draw(board,canvas,ss,coords)

class King(Piece):

	def __init__(self,color,doubleAdvance=None,moved=False):
		super().__init__('king',color,doubleAdvance,moved)

	def getMoves(self,board):
		moves = super().getDirectionMoves(board,['up','down','left','right'] + 
									      ['up-left','up-right','down-left','down-right'],
									      oneMove=True)
		row = 1 if self.color == 'white' else 8
		k_column,k_row = board.getLoc(self)
		for direction,column in zip([-2,2],['a','h']):
			loc = (column,row)
			if (not self.moved and not board.inCheck(self.color) and
				board.squares[loc].hasPiece(self.color) and
				board.squares[loc].getPiece().name == 'rook' and 
				not board.squares[loc].getPiece().moved and
				not any([board.squares[(chr(i),row)].hasPiece()
					 for i in range(ord(column)+1,ord(k_column))])):
				moves.append((chr(ord(k_column)+direction),row))
		return moves

	def move(self,board,square):
		k_column,k_row = board.getLoc(self)
		column,row = square.loc
		if abs(ord(k_column)-ord(column)) == 2: #castle
			direction = 1 if ord(column) < ord(k_column) else - 1
			rook_loc = ('a',row) if direction > 0 else ('h',row)
			rook = board.squares[rook_loc].getPiece()
			board.setPiece(rook,(chr(ord(column)+direction),row))
		super().move(board,square)

	def draw(self,board,canvas,ss):
		coords = ([0.25,0.05,0.75,0.05,0.65,0.35,0.7,0.55,0.7,0.75,0.55,0.75] +
				  [0.55,0.85,0.6,0.85,0.6,0.9,0.55,0.9,0.55,0.95,0.45,0.95] +
				  [0.45,0.9,0.4,0.9,0.4,0.85,0.45,0.85,0.45,0.75] +
				  [0.4,0.75,0.3,0.75,0.3,0.55,0.35,0.35])
		super().draw(board,canvas,ss,coords)

name_dict = {'pawn':Pawn,'rook':Rook,'knight':Knight,
			 'bishop':Bishop,'queen':Queen,'king':King}

