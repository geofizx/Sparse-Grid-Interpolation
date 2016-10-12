#!/usr/bin/env python2.7
# encoding: utf-8

"""
Some unit tests and usage examples for random and sparse grid sampler class

@Usage Examples and Tests

2D Chebyshev sparse grid nodes
2D Clenshaw-Curtis sparse grid nodes
2D Poisson Disk samples
2D Uniform Random samples
2D Stroud/Xiu degree-2, degree-3 nodes

@author Michael Tompkins
@copyright 2016
"""

import matplotlib.pyplot as pl
from code import ndSampler

# Determine which tests will be run with bools
Poisson = False
Chebyshev = False
Clenshaw = False
Uniform = False
Stroud = True

if Poisson is True:
	num = 400		# Number of samples to draw
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(num/2,dim1)
	candidates = 20	# Number of candidate samples for each numsim iteration of sampler
	points1 = sample.poissondisk(candidates)
	sample = ndSampler(num,dim1)
	points2 = sample.poissondisk(candidates)
	label1 = str(num/2)+" Samples"
	label2 = str(num)+" Samples"
	pl.plot(points1[:,0],points1[:,1],'ro',label=label1)
	pl.hold(True)
	pl.plot(points2[:,0],points2[:,1],'bo',label=label2)
	pl.title("Poisson Disk Random Samples in 2-D")
	pl.legend()
	pl.show()

if Chebyshev is True:
	num = 4			# Degree of polynomial to compute
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(num,dim1)
	points1,mi0,indx0 = sample.sparse_sample("CH")
	sample = ndSampler(num/2,dim1)
	points2,mi0,indx0 = sample.sparse_sample("CH")
	label1 = "Degree:"+str(num)+" Nodes"
	label2 = "Degree:"+str(num/2)+" Nodes"
	pl.plot(points1[:,0],points1[:,1],'ro',label=label1)
	pl.hold(True)
	pl.plot(points2[:,0],points2[:,1],'bo',label=label2)
	pl.title("Chebyshev Sparse-Grid Samples in 2-D")
	pl.legend()
	pl.show()

if Clenshaw is True:
	num = 4			# Degree of polynomial to compute
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(num,dim1)
	points1,mi0,indx0 = sample.sparse_sample("CC")
	sample = ndSampler(num/2,dim1)
	points2,mi0,indx0 = sample.sparse_sample("CC")
	label1 = "Degree:"+str(num)+" Nodes"
	label2 = "Degree:"+str(num/2)+" Nodes"
	pl.plot(points1[:,0],points1[:,1],'ro',label=label1)
	pl.hold(True)
	pl.plot(points2[:,0],points2[:,1],'bo',label=label2)
	pl.title("Clenshaw-Curtis Sparse-Grid Samples in 2-D")
	pl.legend()
	pl.show()

if Stroud is True:
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(3,dim1)
	points1 = sample.stroud(3)		# Stroud rule degree-3 points
	label1 = "Stroud Degree:"+str(3)+" Points"
	pl.plot(points1[:,0],points1[:,1],'bo',label=label1)
	points2 = sample.stroud(5)		# Xiu rule degree-3 points
	label2 = "Xiu Degree:"+str(3)+" Points"
	pl.plot(points2[:,0],points2[:,1],'ro',label=label2)
	pl.title("Stroud/Xiu Degree-3 Samples in 2-D")
	pl.legend()
	pl.show()

if Uniform is True:
	num = 400		# Number of samples to draw
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(num/2,dim1)
	points1 = sample.unfrm()
	sample = ndSampler(num,dim1)
	points2 = sample.unfrm()
	label1 = str(num/2)+" Samples"
	label2 = str(num)+" Samples"

	pl.plot(points1[:,0],points1[:,1],'ro',label=label1)
	pl.hold(True)
	pl.plot(points2[:,0],points2[:,1],'bo',label=label2)
	pl.title("Uniform Random Samples in 2-D")
	pl.legend()
	pl.show()