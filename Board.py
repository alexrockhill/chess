from Pieces import Pawn, Rook, Knight, Bishop, Queen, King
from func import (oppositeColor, int2color, loc2int, colors, nameOrder, rowColors)
from importlib import import_module

class Square:

	def __init__(self,column,row,color,piece=None,rect=None,display=None):
		self.loc = (column,row)
		self.color = color
		self.piece = piece
		self.rect = rect
		self.display = display

	def hasPiece(self,color=None):
		return (self.piece is not None if color is None else 
				self.piece is not None and self.piece.color == color)

	def getPiece(self):
		return self.piece

	def removePiece(self):
		if self.piece:
			self.piece.square = None
		self.piece = None

class Board:

	name_dict = {'pawn':Pawn,'rook':Rook,'knight':Knight,
			 'bishop':Bishop,'queen':Queen,'king':King}

	def __init__(self,move=0,moves=None,positions=None,squares=None,pieces=None,
				 ai=None):
		self.move = move  #white even, black odd
		self.moves = [] if moves is None else moves
		self.positions = [] if positions is None else positions
		if squares is None and pieces is None:
			self.makeSquares()
			self.initPieces()
		else:
			self.squares = squares
			self.pieces = pieces
		self.ai = None if ai is None else {color: import_module(ai[color]).AI(color, show=False) for color in ai}

	def initPieces(self):
		self.pieces = {color:{name:[] for name in self.name_dict} for color in colors}
		for name,column in zip(nameOrder,[chr(i) for i in range(97,105)]):
			for color in rowColors:
				self.makePiece('pawn',color,(column,rowColors[color]['pawn']))
				self.makePiece(name,color,(column,rowColors[color]['piece']))
		self.positions.append(self.getPosition())

	def makePiece(self,name,color,loc):
		piece = self.name_dict[name](color,self.squares[loc])
		self.pieces[color][name].append(piece)
		self.squares[loc].piece = piece

	def makeSquares(self):
		self.squares = {}
		for row in range(1,9):
			for column in [chr(i) for i in range(97,105)]:
				color = ('black' if (row%2 and ord(column)%2) or 
								     not (row%2 or ord(column)%2) else 'white')
				self.squares[(column,row)] = Square(column,row,color)

	def makeMove(self,piece,loc):
		self.move += 1
		self.moves.append((piece,loc))
		self.positions.append(self.getPosition())
		self.movePiece(piece,loc)

	def movePiece(self,piece,loc):
		if self.squares[loc].piece:
			self.takePiece(self.squares[loc].piece)
		self.squares[piece.square.loc].piece = None
		self.squares[loc].piece = piece
		piece.move(self,self.squares[loc])

	def getPieces(self,color=None):
		return (self._getPieces(color) if color else 
				[piece for color in colors for piece in self._getPieces(color)])

	def _getPieces(self,color):
		pieces = []
		for name in self.pieces[color]:
			pieces += self.pieces[color][name]
		return pieces

	def takePiece(self,piece):
		self.pieces[piece.color][piece.name].remove(piece)

	def getKingLoc(self,color):
		return self.pieces[color]['king'][0].square.loc

	def getMoves(self,piece):
		return self.checkCheck(piece,piece.getMoves(self))

	def getPosition(self):
		position = {}
		for piece in self.getPieces():
			position[piece] = piece.square.loc
		return position

	def setPosition(self,position):
		for loc in self.squares:
			self.squares[loc].piece = None
		for piece in position:
			self.squares[position[piece]].piece = piece
			piece.square = self.squares[position[piece]]
			if not piece in self.pieces[piece.color][piece.name]: # if it was taken in simulation
				self.pieces[piece.color][piece.name].append(piece)

	def checkCheck(self,piece,moves):
		position = self.getPosition()
		color = piece.color
		other_color = oppositeColor(piece.color)
		for loc in moves.copy():
			self.movePiece(piece,loc)
			if self.inCheck(color):
				moves.remove(loc)
			self.setPosition(position)
		return moves

	def checkCheckMate(self):
		if self.checkRepeatedMoves():
			return 'Draw'
		color = int2color(self.move) # potentially checkmated player's turn
		moves = []
		for piece in self.getPieces(color):
			if len(self.getMoves(piece)) > 0:
				return False
		return 'Check Mate %s' % oppositeColor(color) if self.inCheck(color) else 'Draw'

	def checkRepeatedMoves(self):
		if len(self.positions) > 3:
			if self.positions[-1] == self.positions[-3] and self.positions[-2] == self.positions[-4]:
				return True
		return False

	def inCheck(self,color):
		other_color = oppositeColor(color)
		for piece in self.getPieces(other_color):
			if (piece.name != 'king' and  # king can't give check
				self.getKingLoc(color) in piece.getMoves(self)):
				return True
		return False

	def draw(self,canvas,ss):
		canvas.delete('all')
		for loc in self.squares:
			square = self.squares[loc]
			column,row = loc
			column,row = loc2int(column, row)
			square.rect = canvas.create_rectangle(ss*column,ss*row,ss*column+ss,ss*row+ss,
												  outline='black',fill=square.color,
												  tag='board')
		for piece in self.getPieces():
			piece.draw(canvas,ss)	
		canvas.lower('board')

	def makeAIMove(self):
		self.ai[int2color(self.move)].make_decision(self)

	def isAITurn(self):
		return int2color(self.move) in self.ai

	def wasAITurn(self):
		return int2color(self.move - 1) in self.ai

	def getAIPromotion(self):
		return self.ai[int2color(self.move)].get_promotion(self, loc)







