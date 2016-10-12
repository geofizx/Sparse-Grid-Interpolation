#!/usr/bin/env python2.7
# encoding: utf-8

"""
Some unit tests and usage examples for Smolyak Sparse Grid Interpolation

@Usage Examples and Tests

2D Chebyshev polynomial sparse grid interpolation of 2D test function in fun_nd
2D Clenshaw-Curtis piece-wise linear basis sparse grid interpolation of 2D test function in fun_nd

Generate plots for some outputs

@author Michael Tompkins
@copyright 2016
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as npy
import matplotlib.pyplot as pl
from code import sparseInterp, fun_nd

# Determine which tests will be run with bools
Chebyshev = True
Clenshaw = True

if Chebyshev is True:

	# Run Chebyshev polynomial sparse-grid interpolation of 2D test function in fun_nd
	type1 = "CH"

	n = 6	# Maximum degree of interpolation to consider - early stopping may use less degree exactness
	dim1 = 2	# Dimensionality of function to interpolate
	gridout = npy.asarray([[0.0,0.25,0.5,0.75,1.0],[0.0,0.25,0.5,0.75,1.0]]).T
	[xx,yy] = npy.meshgrid([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	gridout = npy.asarray([xx.reshape(121),yy.reshape(121)]).T
	intval = npy.asarray([[0.0,1.0],[0.0,1.0]]).T

	# Instantiate and run interpolation for Chebyshev Polynomials
	interp = sparseInterp(n, dim1, gridout, type1, intval)
	output,meanerr1,werr1 = interp.runInterp()

	# Compare results with true function
	tmpvals = npy.asarray(fun_nd.fun_nd(gridout))
	tmpval2 = tmpvals.reshape(11,11)

	fig = pl.figure()
	pl.title("Chebyshev Polynomial Basis")
	ax = fig.add_subplot(131, projection='3d',title="True Function")
	ax.plot_surface(xx, yy, tmpval2,  rstride=1, cstride=1, cmap=cm.jet)
	ax = fig.add_subplot(132, projection='3d',title="Interpolation")
	tmpval3 = output.reshape(11,11)
	ax.plot_surface(xx, yy, tmpval3,  rstride=1, cstride=1, cmap=cm.jet)
	ax = fig.add_subplot(133, projection='3d', title="Interpolation Error")
	tmpval4 = npy.abs(tmpval3 - tmpval2)
	ax.plot_surface(xx, yy, tmpval4,  rstride=1, cstride=1, cmap=cm.jet)
	ax.set_zlim(0.0,npy.max(tmpval4)*2)
	pl.show()

	print "Mean Error for Each Degree of Total Degree:",n,": ",meanerr1

if Clenshaw is True:

	# Run Clenshaw-Curtis Piece-wise linear sparse-grid Interpolation of 2D test function in fun_nd
	type1 = "CC"

	n = 6	# Maximum degree of interpolation to consider - early stopping may use less degree exactness
	dim1 = 2	# Dimensionality of function to interpolate
	gridout = npy.asarray([[0.0,0.25,0.5,0.75,1.0],[0.0,0.25,0.5,0.75,1.0]]).T
	[xx,yy] = npy.meshgrid([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	gridout = npy.asarray([xx.reshape(121),yy.reshape(121)]).T
	intval = npy.asarray([[0.0,1.0],[0.0,1.0]]).T

	# Instantiate and run interpolation for Chebyshev Polynomials
	interp = sparseInterp(n, dim1, gridout, type1, intval)
	output,meanerr1,werr1 = interp.runInterp()

	# Compare results with true function
	tmpvals = npy.asarray(fun_nd.fun_nd(gridout))
	tmpval2 = tmpvals.reshape(11,11)

	fig = pl.figure()
	pl.title("Clenshaw-Curtis Piece-Wise Linear Basis")
	ax = fig.add_subplot(131, projection='3d',title="True Function")
	ax.plot_surface(xx, yy, tmpval2,  rstride=1, cstride=1, cmap=cm.jet)
	ax = fig.add_subplot(132, projection='3d',title="Interpolation")
	tmpval3 = output.reshape(11,11)
	ax.plot_surface(xx, yy, tmpval3,  rstride=1, cstride=1, cmap=cm.jet)
	ax = fig.add_subplot(133, projection='3d', title="Interpolation Error")
	tmpval4 = npy.abs(tmpval3 - tmpval2)
	ax.plot_surface(xx, yy, tmpval4,  rstride=1, cstride=1, cmap=cm.jet)
	ax.set_zlim(0.0,npy.max(tmpval4)*2)
	pl.show()

	print "Mean Error for Each Degree of Total Degree:",n,": ",meanerr1
