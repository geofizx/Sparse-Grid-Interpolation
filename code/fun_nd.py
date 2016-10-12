#!/usr/bin/env python2.7
# encoding: utf-8

"""
@description

Example function evaluation function for use with sparse_vals class
Perform some simple 2D function evaluations for running sparse_vals interpolation tests

@author Michael Tompkins
@copyright 2016
"""

import numpy as npy

def fun_nd(x):

	if x.shape[1] == 2:	# Run 2D test function evaluations
		func2d = []
		for i in range(x.shape[0]):
			func2d.append((0.5/npy.pi*x[i,0]-.51/(.4*npy.pi**2)*x[i,0]**2+x[i,1]-(.6))**2 + \
						  1*(1-1/(.8*npy.pi))*npy.cos(x[i,0])+.10)

		return func2d

	else:
		pass
		return None

if __name__ == "__main__":

	"""
	Unit test
	"""
	x = npy.asarray([[0.,0.5,1.0],[0.0,0.5,1.0]]).T
	func_out = fun_nd(x)
	print func_out