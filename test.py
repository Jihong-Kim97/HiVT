from Bezier import Bezier
import numpy as np
t_points = np.arange(0, 1, 0.01) #................................. Creates an iterable list from 0 to 1.
points1 = np.array([[0, 0], [0, 8], [5, 10], [9, 7], [4, 3]]) #.... Creates an array of coordinates.
curve1 = Bezier.Curve(t_points, points1)
print(curve1[:,0])