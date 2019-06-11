from func import oppositeColor, color2int, moveloc, loc2int, isPawnStartingRow, pieceValues

class Piece(object):

	def __init__(self,name,color,square,doubleAdvance=None,moved=None,display=None):
		self.name = name
		self.color = color
		self.square = square
		self.doubleAdvance = doubleAdvance
		self.moved = moved
		self.display = display
		self.value = pieceValues(name)

	def getDirectionMoves(self,board,directions,oneMove=False):
		moves = []
		for direction in directions:
			loc = moveloc(self.square.loc,direction)
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
		pass

	def move(self,board,square):
		self.square = square

	def draw(self,canvas,ss,coords):
		colorint = color2int(self.color)
		colorint2 = self.color=='white'
		column,row = self.square.loc
		column,row = loc2int(column,row)
		coords = [(colorint2-c)*colorint for c in coords]
		coords = [ss*row + ss*c if i%2 else ss*column+ ss*c for i,c in enumerate(coords)]
		self.display = canvas.create_polygon(coords,fill=self.color,
											 outline=oppositeColor(self.color),
											 tag='piece')
class Pawn(Piece):

	def __init__(self,color,square,doubleAdvance=0,moved=None):
		super().__init__('pawn',color,square,doubleAdvance,moved)

	def getMoves(self,board):
		moves = []
		colorint = color2int(self.color)
		column,row = self.square.loc
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
		if isPawnStartingRow(self.color,self.square.loc):
			advanceloc2 = (column, row+colorint*2)
			if not (board.squares[advanceloc2].hasPiece() or
					board.squares[advanceloc].hasPiece()):
				moves.append(advanceloc2)			
		return moves

	def move(self,board,square):
		column,row = square.loc
		my_column,my_row = self.square.loc
		if abs(row - my_row) == 2:
			self.doubleAdvance = board.move # for en passant
		if column != my_column and not square.hasPiece(): # take and no piece there
			board.takePiece(board.squares[(column,my_row)].piece)
		super().move(board,square)

	def draw(self,canvas,ss):
		coords = ([0.3,0.2,0.7,0.2,0.6,0.5,0.65,0.52,0.67,0.57,0.67,0.62,0.65,0.64] + 
				  [0.6,0.65,0.5,0.67,0.4,0.65,0.35,0.64,0.33,0.62,0.33,0.57,0.35,0.52] +
				  [0.4,0.5])
		super().draw(canvas,ss,coords)

class Rook(Piece):

	def __init__(self,color,square,doubleAdvance=None,moved=False):
		super().__init__('rook',color,square,doubleAdvance,moved)

	def getMoves(self,board):
		return super().getDirectionMoves(board,['up','down','left','right'])

	def move(self,board,square):
		self.moved = True
		super().move(board,square)

	def draw(self,canvas,ss):
		coords = ([0.25,0.2,0.75,0.2,0.75,0.7,0.65,0.7,0.65,0.6,0.55,0.6] +
				  [0.55,0.7,0.45,0.7,0.45,0.6,0.35,0.6,0.35,0.7,0.25,0.7])
		super().draw(canvas,ss,coords)

class Knight(Piece):

	def __init__(self,color,square,doubleAdvance=None,moved=None):
		super().__init__('knight',color,square,doubleAdvance,moved)

	def getMoves(self,board):
		moves = []
		row, column = self.square.loc
		for firstDirection in ['up','down','left','right']:
			if firstDirection in ['up','down']:
				secondDirections = ['left','right']
			elif firstDirection in ['left','right']:
				secondDirections = ['up','down']
			for secondDirection in secondDirections:
				loc = moveloc(self.square.loc,firstDirection)
				loc = moveloc(loc,secondDirection)
				loc = moveloc(loc,secondDirection)
				if (loc in board.squares and 
					(not board.squares[loc].hasPiece() or
					 board.squares[loc].hasPiece(oppositeColor(self.color)))):
					moves.append(loc)
		return moves

	def draw(self,canvas,ss):
		coords = ([0.25,0.2,0.65,0.2,0.55,0.5,0.65,0.6,0.75,0.5,0.75,0.6] +
				  [0.65,0.8,0.55,0.75,0.45,0.7,0.35,0.65,0.25,0.6])
		super().draw(canvas,ss,coords)

class Bishop(Piece):

	def __init__(self,color,square,doubleAdvance=None,moved=None):
		super().__init__('bishop',color,square,doubleAdvance=doubleAdvance,moved=moved)

	def getMoves(self,board):
		return super().getDirectionMoves(board,['up-left','up-right',
												'down-left','down-right'])

	def draw(self,canvas,ss):
		coords = ([0.35,0.2,0.65,0.2,0.55,0.4,0.7,0.6,0.5,0.8] +
				  [0.3,0.6,0.45,0.4])
		return super().draw(canvas,ss,coords)

class Queen(Piece):

	def __init__(self,color,square,doubleAdvance=None,moved=None):
		super().__init__('queen',color,square,doubleAdvance,moved)

	def getMoves(self,board):
		return super().getDirectionMoves(board,['up','down','left','right',
												'up-left','up-right','down-left',
												'down-right'])

	def draw(self,canvas,ss):
		coords = ([0.25,0.1,0.75,0.1,0.65,0.4,0.7,0.6,0.7,0.8,0.6,0.8] +
				  [0.6,0.9,0.4,0.9,0.4,0.8,0.3,0.8,0.3,0.6,0.35,0.4])
		super().draw(canvas,ss,coords)

class King(Piece):

	def __init__(self,color,square,doubleAdvance=None,moved=False):
		super().__init__('king',color,square,doubleAdvance,moved)

	def getMoves(self,board):
		moves = super().getDirectionMoves(board,['up','down','left','right'] + 
									      ['up-left','up-right','down-left','down-right'],
									      oneMove=True)
		row = 1 if self.color == 'white' else 8
		k_column,k_row = self.square.loc
		for direction,column in zip([-2,2],['a','h']):
			loc = (column,row)
			if (not self.moved and
				board.squares[loc].hasPiece(self.color) and
				board.squares[loc].getPiece().name == 'rook' and 
				not board.squares[loc].getPiece().moved and
				not any([board.squares[(chr(i),row)].hasPiece()
					 for i in range(ord(column)+1,ord(k_column))])):
				moves.append((chr(ord(k_column)+direction),row))
		return moves

	def move(self,board,square):
		self.moved = True
		k_column,k_row = self.square.loc
		column,row = square.loc
		if abs(ord(k_column)-ord(column)) == 2: #castle
			direction = 1 if ord(column) < ord(k_column) else - 1
			rook_loc = ('a',row) if direction > 0 else ('h',row)
			rook = board.squares[rook_loc].getPiece()
			board.movePiece(rook,board.squares[(chr(ord(column)+direction),row)].loc)
		super().move(board,square)

	def draw(self,canvas,ss):
		coords = ([0.25,0.05,0.75,0.05,0.65,0.35,0.7,0.55,0.7,0.75,0.55,0.75] +
				  [0.55,0.85,0.6,0.85,0.6,0.9,0.55,0.9,0.55,0.95,0.45,0.95] +
				  [0.45,0.9,0.4,0.9,0.4,0.85,0.45,0.85,0.45,0.75] +
				  [0.4,0.75,0.3,0.75,0.3,0.55,0.35,0.35])
		super().draw(canvas,ss,coords)

