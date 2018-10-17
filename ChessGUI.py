from tkinter import Tk, Canvas, Frame
from Board import Board, name_dict
from func import oppositeColor, int2loc, int2color, isLastRank
from PIL import ImageGrab

class ChessGui(Frame):

	def __init__(self, root):
		self.root = root
		Frame.__init__(self, self.root)
		self.board = Board()
		width = self.root.winfo_screenwidth()
		height = self.root.winfo_screenheight()
		self.size = min([height,width])
		self.canvas = Canvas(self.root,width=self.size,height=self.size)
		self.squareSize = 0.75*min([height,width])/8
		self.board.draw(self.canvas,self.squareSize)
		self.canvas.pack(fill='both', expand=True)
		self._drag_data = {'loc':(0,0),'loc_i':(0,0),'loc_offset':(0,0),
						   'piece_loc':None}
		self.canvas.tag_bind('piece', '<ButtonPress-1>', self.on_piece_press)
		self.canvas.tag_bind('piece', '<ButtonRelease-1>', self.on_piece_release)
		self.canvas.tag_bind('piece', '<B1-Motion>', self.on_piece_motion)

		'''self.canvas.update()
		x = root.winfo_rootx()
		y = root.winfo_rooty()
		ImageGrab.grab((x,y,x+self.size*1.7,y+self.size*1.7)).save('example.jpg')'''

	def loc2piece(self):
		return self.board.squares[self._drag_data['piece_loc']].getPiece()

	def _find_square_loc(self,loc):
		x,y = loc
		x = int(round((x-self.squareSize/2)/self.squareSize))
		y = int(round((y-self.squareSize/2)/self.squareSize))
		loc = int2loc(x,y)
		square = self.board.squares[loc]
		x1,y1,x2,y2 = self.canvas.coords(square.rect)
		x,y = (x1+x2)/2,(y1+y2)/2
		return square,(x,y)

	def on_piece_press(self, event):
		item = self.canvas.find_closest(event.x,event.y)[0]
		if 'piece' in self.canvas.gettags(item):
			square,loc = self._find_square_loc((event.x,event.y))
			x,y = loc
			piece = square.getPiece()
			if (self.board.move is not None and piece is not None and
				piece.color == int2color(self.board.move)):
				self._drag_data['loc'] = (event.x,event.y)
				self._drag_data['loc_i'] = loc
				self._drag_data['loc_offset'] = (x-event.x,y-event.y)
				self._drag_data['piece_loc'] = piece.square.loc
				self.canvas.lift(piece.display)

	def on_piece_release(self, event):
		if self._drag_data['piece_loc'] is not None:
			square,targ_loc = self._find_square_loc((event.x,event.y))
			targ_x,targ_y = targ_loc
			ok = square.loc in self.board.getMoves(self.loc2piece())
			if not ok:
				targ_x,targ_y = self._drag_data['loc_i']
			offset_x,offset_y = self._drag_data['loc_offset']
			delta_x = targ_x - offset_x - event.x
			delta_y = targ_y - offset_y - event.y
			self.canvas.move(self.loc2piece().display, delta_x, delta_y)
			if ok:
				piece = square.getPiece()
				if piece is not None:
					self.canvas.delete(piece.display)
				self.board.makeMove(self.loc2piece(),square.loc)
				self._drag_data['piece_loc'] = square.loc
				if (self.loc2piece().name == 'pawn' and 
					isLastRank(self.loc2piece().color,square.loc)):
					name = None
					while name is None:
						name = input('What to promote pawn to?\t').lower()
						if name in name_dict and not name in ['pawn','king']:
							self.board._makePiece(name,self.loc2piece().color,
												  square.loc)

						else:
							name = None
				outcome = self.board.checkCheckMate()
				if outcome:
					print(outcome)
					self.board.move = None
			self.board.draw(self.canvas,self.squareSize)
			self._drag_data['piece_loc'] = None
			self._drag_data['loc'] = (0,0)

	def on_piece_motion(self,event):
		if self._drag_data['piece_loc'] is not None:
			x,y = self._drag_data['loc']
			self.canvas.move(self.loc2piece().display,event.x-x,event.y-y)
			self._drag_data['loc'] = (event.x,event.y)
		
if __name__ == '__main__':
	root = Tk()
	CG = ChessGui(root)
	root.mainloop()

    


