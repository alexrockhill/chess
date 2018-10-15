from Pieces import Pawn, Rook, Knight, Bishop, Queen, King, name_dict
from func import (oppositeColor, color2int, int2color, moveloc, loc2int, int2loc,
				  makePiece, isPawnStartingRow, isLastRank, colors)
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
		if self.piece is not None:
			self.piece.square = None
		self.piece = None

class Board:

	def __init__(self,move=0,moves=None,positions=None,squares=None,pieces=None):
		self.move = move #white even, black odd
		self.moves = [] if moves is None else moves
		self.positions = [] if positions is None else positions
		if squares is None and pieces is None:
			self._makeSquares()
			self._initPieces()
		else:
			self.squares = squares
			self.pieces = pieces

	def _initPieces(self):
		self.pieces = {'white':{name:{} for name in name_dict},
					   'black':{name:{} for name in name_dict}}
		rowColors = {'white':{'piece':1,'pawn':2},'black':{'piece':8,'pawn':7}}
		nameOrder = ['rook','knight','bishop','queen',
					  'king','bishop','knight','rook']
		for name,column in zip(nameOrder,[chr(i) for i in range(97,105)]):
			for color in rowColors:
				self._makePiece('pawn',color,(column,rowColors[color]['pawn']))
				self._makePiece(name,color,(column,rowColors[color]['piece']))
		self.positions.append(self.getPosition())

	def _makePiece(self,name,color,loc):
		piece = name_dict[name](color,self.squares[loc])
		self.pieces[color][name][piece] = loc
		self.squares[loc].piece = piece

	def _makeSquares(self):
		self.squares = {}
		for row in range(1,9):
			for column in [chr(i) for i in range(97,105)]:
				color = ('black' if (row%2 and ord(column)%2) or 
								     not (row%2 or ord(column)%2) else 'white')
				self.squares[(column,row)] = Square(column,row,color)

	def makeMove(self,piece,loc):
		self.move += 1
		self.moves.append((piece.name,loc))
		self.positions.append(self.getPosition())
		piece.move(self,self.squares[loc])
		self.takePiece(loc)
		self.setPiece(piece,loc)
		
	def takePiece(self,loc):
		if self.squares[loc].piece:
			piece = self.squares[loc].piece
			self.pieces[piece.color][piece.name][piece] = None
			self.squares[loc].piece = None

	def setPiece(self,piece,loc):
		if self.pieces[piece.color][piece.name][piece] is not None:
			self.squares[self.pieces[piece.color][piece.name][piece]].piece = None
		self.squares[loc].piece = piece
		self.pieces[piece.color][piece.name][piece] = loc

	def getLoc(self,piece):
		return self.pieces[piece.color][piece.name][piece]

	def getKingLoc(self,color):
		return list(self.pieces[color]['king'].values())[0]

	def getMoves(self,piece):
		return self.checkCheck(piece,piece.getMoves(self))

	def getPosition(self):
		position = {}
		for color in self.pieces:
			for name in self.pieces[color]:
				for piece in self.pieces[color][name]:
					position[piece] = self.pieces[color][name][piece]
		return position

	def setPosition(self,position):
		for loc in self.squares:
			self.squares[loc].piece = None
		for piece in position:
			loc = position[piece]
			if loc is not None:
				self.setPiece(piece,loc)

	def checkCheck(self,piece,moves):
		position = self.getPosition()
		color = piece.color
		other_color = oppositeColor(piece.color)
		for loc in moves.copy():
			self.setPiece(piece,loc)
			for name in self.pieces[other_color]:
				for other_piece in self.pieces[other_color][name]:
					if (self.pieces[other_color][name][other_piece] is not None and
						self.getKingLoc(color) in other_piece.getMoves(self) and
						loc in moves and loc != self.getLoc(other_piece)):
						moves.remove(loc)
			self.setPosition(position)
		return moves

	def checkCheckMate(self):
		color = int2color(self.move) # potentially checkmated player's turn
		moves = []
		for name in self.pieces[color]:
			for piece in self.pieces[color][name]:
				if (self.pieces[color][name][piece] is not None and 
					len(self.getMoves(piece)) > 0):
					return False
		return 'Check Mate %s' %(oppositeColor(color)) if self.inCheck(color) else 'Draw'

	def inCheck(self,color):
		other_color = oppositeColor(color)
		for name in self.pieces[other_color]:
			for piece in self.pieces[other_color][name]:
				if (name != 'king' and self.pieces[other_color][name][piece] is not None and
					self.getKingLoc(color) in piece.getMoves(self)):
					return True
		return False

	def draw(self,canvas,ss):
		canvas.delete('all')
		for loc in self.squares:
			square = self.squares[loc]
			column,row = loc
			column,row = loc2int(column,row)
			square.rect = canvas.create_rectangle(ss*column,ss*row,ss*column+ss,ss*row+ss,
												  outline='black',fill=square.color,
												  tag='board')
		for color in colors:
			for name in self.pieces[color]:
				for piece in self.pieces[color][name]:
					if self.pieces[color][name][piece] is not None:
						piece.draw(self,canvas,ss)	
		canvas.lower('board')






