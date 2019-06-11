colors = ['white','black']
rowColors = {'white':{'piece':1,'pawn':2},'black':{'piece':8,'pawn':7}}
nameOrder = ['rook','knight','bishop','queen',
			  'king','bishop','knight','rook']

piece_values = {'pawn':1,'knight':3,'bishop':3,'rook':5,'queen':9,'king':0}

def pieceValues(name):
	if name in piece_values:
		return piece_values[name]
	raise ValueError('Name not recognized')

def color2int(color):
	if not color in ['white','black']:
		raise ValueError('Unrecognized color')
	return (color == 'white') - (color == 'black')

def int2color(i):
	return 'black' if i % 2 else 'white'

def moveloc(loc,direction):
	column,row = loc
	if direction == 'up':
		return (column,row+1)
	elif direction == 'down':
		return (column,row-1)
	elif direction == 'left':
		return (chr(ord(column)-1),row)
	elif direction == 'right':
		return (chr(ord(column)+1),row)
	elif direction == 'up-left':
		return (chr(ord(column)-1),row+1)
	elif direction == 'up-right':
		return (chr(ord(column)+1),row+1)
	elif direction == 'down-left':
		return (chr(ord(column)-1),row-1)
	elif direction == 'down-right':
		return (chr(ord(column)+1),row-1)
	else:
		raise ValueError('Unrecognized direction')

def oppositeColor(color):
	if color == 'black':
		return 'white' 
	elif color == 'white':
		return 'black'
	else:
		raise ValueError('Unrecognized color')

def loc2int(column,row):
	return ord(column)-97,8-row

def int2loc(x,y):
	return chr(x+97),8-y

def isPawnStartingRow(color,loc):
	column,row = loc
	if color == 'white':
		return row == 2
	elif color == 'black':
		return row == 7
	else:
		raise ValueError('Color error: color must be white or black')

def isLastRank(color,loc):
	column,row = loc
	if color == 'white':
		return row == 8
	elif color == 'black':
		return row == 1
	else:
		raise ValueError('Color error: color must be white or black')