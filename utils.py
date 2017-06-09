from math import sqrt

def distance(a, b):
	'''
		Parameter: a, b are two tuples in form of (x, y) that contains the coordinate of
		two locations

		Return: The distance between a and b
	'''
	return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

