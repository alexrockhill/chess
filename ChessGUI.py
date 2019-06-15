from tkinter import Tk, Canvas, Frame, simpledialog, messagebox
from Board import Board
from AI import AI, Network, Node, ConnectionWeight
from func import opposite_color, int2loc, int2color, is_last_rank
from PIL import ImageGrab
import time


class ChessGui(Frame):

	def __init__(self, root):
		self.root = root
		Frame.__init__(self, self.root)
		self.board = Board() 
		width = self.root.winfo_screenwidth()
		height = self.root.winfo_screenheight()
		self.size = min([height,width])
		self.canvas = Canvas(self.root, width=self.size, height=self.size)
		self.squareSize = 0.75*min([height, width])/8
		self.board.draw(self.canvas,self.squareSize)
		self.canvas.pack(fill='both', expand=True)
		self._drag_data = {'loc': (0, 0), 'loc_i': (0, 0), 'loc_offset': (0, 0),
						   'piece_loc': None}
		self.canvas.tag_bind('piece', '<ButtonPress-1>', self.on_piece_press)
		self.canvas.tag_bind('piece', '<ButtonRelease-1>', self.on_piece_release)
		self.canvas.tag_bind('piece', '<B1-Motion>', self.on_piece_motion)
		self.ai = {color: AI(color, show=False) for color in ['white', 'black']}
		while not self.board.game_over and self.is_ai_turn():
			self.make_ai_move()
			self.check_promotion_or_game_end()
			self.board.draw(self.canvas, self.squareSize)
			self.root.update()
			time.sleep(1)

		'''self.canvas.update()
		x = root.winfo_rootx()
		y = root.winfo_rooty()
		ImageGrab.grab((x,y,x+self.size*1.7,y+self.size*1.7)).save('example.jpg')'''

	def make_ai_move(self):
		self.ai[int2color(self.board.move)].make_decision(self.board)

	def is_ai_turn(self):
		return self.ai is not None and int2color(self.board.move) in self.ai

	def was_ai_turn(self):
		return self.ai is not None and int2color(self.board.move - 1) in self.ai

	def loc2piece(self):
		return self.board.squares[self._drag_data['piece_loc']].getPiece()

	def _find_square_loc(self,loc):
		x, y = loc
		x = int(round((x-self.squareSize/2)/self.squareSize))
		y = int(round((y-self.squareSize/2)/self.squareSize))
		loc = int2loc(x, y)
		square = self.board.squares[loc]
		x1, y1, x2, y2 = self.canvas.coords(square.rect)
		x, y = (x1 + x2) / 2, (y1 + y2) / 2
		return square, (x, y)

	def on_piece_press(self, event):
		item = self.canvas.find_closest(event.x,event.y)[0]
		if 'piece' in self.canvas.gettags(item):
			square, loc = self._find_square_loc((event.x,event.y))
			x, y = loc
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
			square, targ_loc = self._find_square_loc((event.x, event.y))
			targ_x, targ_y = targ_loc
			ok = square.loc in self.board.get_moves(self.loc2piece())
			if not ok:
				targ_x, targ_y = self._drag_data['loc_i']
			offset_x, offset_y = self._drag_data['loc_offset']
			delta_x = targ_x - offset_x - event.x
			delta_y = targ_y - offset_y - event.y
			self.canvas.move(self.loc2piece().display, delta_x, delta_y)
			if ok:
				piece = square.getPiece()
				if piece is not None:
					self.canvas.delete(piece.display)
				self.board.make_move(self.loc2piece(), square.loc)
				self._drag_data['piece_loc'] = square.loc
				self.check_promotion_or_game_end()
			if self.board.move is not None and self.is_ai_turn():
				self.make_ai_move()
			self.board.draw(self.canvas, self.squareSize)
			self._drag_data['piece_loc'] = None
			self._drag_data['loc'] = (0, 0)

	def on_piece_motion(self, event):
		if self._drag_data['piece_loc'] is not None:
			x,y = self._drag_data['loc']
			self.canvas.move(self.loc2piece().display, event.x-x, event.y-y)
			self._drag_data['loc'] = (event.x, event.y)

	def check_promotion_or_game_end(self):
		piece, loc = self.board.moves[-1]
		if piece.name == 'pawn' and is_last_rank(int2color(self.board.move - 1), loc):
			if not self.was_ai_turn():
				name = None
				while name is None:
					name = simpledialog.askstring('Pawn Promotion',
												  'What to promote pawn to?\n' +
												  '(queen, knight, bishop or rook)',
												   parent=self.root)
					if name not in Board.name_dict or name in ['pawn', 'king']:
						name = None
				self.canvas.delete(piece)  # remove pawn
				self.board.take_piece(piece)
				self.board.make_piece(name, piece.color, loc)
		outcome = self.board.check_check_mate()
		if outcome:
			messagebox.showinfo('Game Over', outcome)


if __name__ == '__main__':
	root = Tk()
	CG = ChessGui(root)
	root.mainloop()

