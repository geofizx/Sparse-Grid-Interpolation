#!/usr/bin/env python2.7
# encoding: utf-8

"""
A general utility class for sampling points in d-dimensions based on Smolyak's sparse grid algorithm using
Clenshaw-Curtis or Chebyshev Polynomial Roots. Also contains utility methods for Poisson Disk, or Uniform Random Draws

@author Michael Tompkins
@copyright 2016
"""

import numpy as npy
import itertools as it
import random
from numpy.matlib import repmat
import math

class ndSampler:

	def __init__(self, samples, dimensions):

		"""
		:arg samples : integer number of points to draw (poisson or unfrm) or degree of polynomial interpolation (Cheby only)
		:arg d : integer dimensionality of sampling space

		options - verbose : turn on (True) or off (False) print statements
		"""

		self.n = samples
		self.d = dimensions
		self.verbose = False

	def sparse_sample(self,type):

		"""
		Sparse-Grid - Polynomial Root Node Enumeration

		:arg : type : string type of polyomial to use for root nodes ("CH" - Chebyshev, "CC" - Clenshaw Curtis)
		:return : Y : array-type (numsim,d) of poynomial nodal points in d-dimensions
		:return : mi : list specifying the number of samples for each degree (level) of samples
		:return : indxi3 : array-type (numsim,d) - ordered index sets [indxi3] for combinations of tensor products
		"""

		# Check type input and exit if not supported
		type_s = ["CH","ch","cc","CC"]
		if type not in type_s:
			print "input type String must be 'CH' or 'CC'"
			exit(1)

		# Set local sampling parameters
		n = self.n
		d = self.d

		if n == 0:
			lp = 0
			Y = npy.ones(shape=(1,d),dtype=float)
			Y *= 0.5
			indx_out = npy.asarray([0,0])
			mi = [1]
		else:
			lp = n
			q = lp + d

			#Next call sparse cartesian product function to compute multi-index in dim=d from 1D index [indxi]
			if self.verbose is True:
				print "..............Computing Multi-Indices for Sparse Grids"

			indxi3 = self.getseq()

			#Now compute number of nodes [mi] and node coordinates [xi] for each value
			#By definition: for i=1, mi(1) = 1; xi(1,1) = 0.5
			mi = []
			mi.append(1)
			xi = [0.5]
			for i in range(2,(q-d+1)+1):
				mi.append(2**(i-1)+1)
				xit = []
				for j in range(1,mi[i-1]+1):
					if type == "CH" or type == "ch":
						xit.append((1+(-(npy.cos((npy.pi*(j-1))/(mi[i-1]-1)))))/2.0)
					else:
						xit.append(float(j-1)/(mi[i-1]-1))
				xi.append(xit)

			pnt = npy.ndarray(shape=(100000,d),dtype=float)

			if self.verbose is True:
				print "..............Computing Cartesian Products for Sparse Grid Nodes"

			tt = 0
			indxi4 = []
			for i in range(0,len(indxi3[:,0])):
				dim = 1
				for p in range(0,d):
					dim = dim * mi[indxi3[i,p]]

				pnt_t = npy.ndarray(shape=(dim,d),dtype = float)
				indxt = repmat(indxi3[i,:],pnt_t.shape[0],1)
				indxi4.extend(indxt)
				for j in range(0,len(indxi3[0,:])):
					xt = npy.asarray(xi[indxi3[i,j]])

					tmp2 = npy.ndarray(shape=(dim,1),dtype=float)
					tmp = npy.ndarray(shape=(dim,1),dtype=float)

					for k in range(0,dim,mi[indxi3[i,j]]):
						tmp[k:k + mi[indxi3[i,j]],0] = xt
					if i == 0:
						inc = 1
					else:
						inc = 1
						for m in range(0,j):
							inc = inc * mi[indxi3[i,m]]
					for k in range(0,dim,inc):
						kstart = k/inc
						kend = kstart + inc * mi[indxi3[i,j]]
						if inc == 1:
							tmp2[:,0] = tmp[:,0]
						else:
							tmp2[k:k+inc,0] = tmp[kstart:kend:mi[indxi3[i,j]],0]

					pnt_t[:,j] = tmp2[:,0]

					# Now pack temporary grids into final grid
					pnt[tt:(tt + len(pnt_t[:,0])),0:d] = pnt_t

				tt += len(pnt_t[:,0])

			pnt2 = npy.round(pnt[0:tt,:],decimals=5)
			indxi4 = npy.asarray(indxi4)

			#Check for redundancies and remove and assign unique rows to [pnt3]
			if self.verbose is True:
				print "..............Checking for Redundant Grid Nodes"

			#Convert numpy array to list of strings for efficient redundancy check
			hashtab = []
			for i in range(0,len(pnt2[:,0])):
				hashtab.append(str(pnt2[i,:]))

			#Build list for compiling unique entries
			hashtab2 = []
			indx_map = []
			for r,x in enumerate(hashtab):
				if x not in hashtab2:
					hashtab2.append(x)
					indx_map.append(r)

			#Now convert strings back to numpy array for return and/or output
			if self.verbose is True:
				print "..............Performing Final Grid Conversion"

			Y = npy.ndarray(shape=(len(hashtab2),d),dtype=float)
			indx_out = npy.ndarray(shape=(len(hashtab2),d),dtype=int)
			ct1 = 0
			for r,x in enumerate(hashtab2):
				ct2 = 0
				desc = x.strip("[")
				desc = desc.strip("]")
				desc = desc.split()
				indx_out[ct1,:] = indxi4[indx_map[r],:]
				for i in desc:
					Y[ct1,ct2] = float(i)
					ct2 += 1
				ct1 += 1

		return Y, mi, indx_out

	def poissonpoly(self,polytope,radius):

		"""
		Resample and add points to polytope defined by samples in argument
		"""

		# Set local sampling parameters
		numsim = self.n
		d = self.d

		bestVal = 0
		numCandidates = min([20,len(polytope[:,0])])
		c = npy.zeros(shape=d,dtype=float)

		Y = npy.ndarray(shape=(numsim,d),dtype=float)

		for i in range(0,numsim):

			t = npy.random.RandomState()

			bestDistance = -1
			#Draw random samples from polytope only
			indexlist = random.sample(xrange(0,len(polytope[:,0])),numCandidates)

			for j in range(0,numCandidates):

				for k in range(0,d):
					c[k] = polytope[indexlist[j],k] + t.uniform(-radius,radius,1)

				dist = npy.min(npy.sqrt(npy.sum(((polytope-c)**2),axis=1)))
				if dist > bestDistance:
					bestDistance = dist
					bestVal = c

			Y[i,:] = bestVal

		return Y

	def poissondisk(self,num_candidates=20):

		"""
		Sampler based on Poisson disk random sampling in d-dimensions over domain [0,1]

		:arg num_candidates : integer number of candidates to draw for each of numsim iterations

		Note: runtime increases with increasing numCandidates to draw for each of numsim iterations, however, the success
		of this method in optimally distributing samples also increases with increasing numCandidates
		"""

		# Set local sampling parameters
		numsim = self.n
		d = self.d

		bestVal = 0
		p = npy.random.RandomState()

		Y = npy.ndarray(shape=(numsim,d),dtype=float)
		c = npy.ndarray(shape=(num_candidates,d),dtype=float)

		#Draw first sample at random in domain [0,1] over d-dimensions
		for k in range(0,d):
			c[0,k] = p.uniform(0,1,1)
		Y[0,:] = c[0,:]

		for i in range(1,numsim):		# Iterate over numsim desired points in d-dimensions

			t = npy.random.RandomState()

			bestDistance = -1
			#Draw candidate uniform random samples from bounded domain
			for k in range(0,d):
				c[:,k] = t.uniform(0,1,num_candidates)

			for j in range(0,num_candidates):	# Test candidates for best distribution about current set

				# Find closest point to current set
				dist = npy.min(npy.sqrt(npy.sum(((Y[0:i,:]-c[j,:])**2),axis=1)))
				if dist > bestDistance:
					bestDistance = dist
					bestVal = c[j,:]

			Y[i,:] = bestVal

		return Y

	def unfrm(self):

		"""
		Sampler based on uniform random sampling of numsim points in d-dimensions over domain [0,1]
		"""
		# Set local sampling parameters
		numsim = self.n
		d = self.d

		#Instantiate Generator for proper seeding
		t = npy.random.RandomState()

		Y = npy.ndarray(shape=(numsim,d),dtype=float)

		#Draw random samples from bounded domain
		for i in range(0,d):
			Y[:,i] = t.uniform(0,1,numsim)

		return Y


	def stroud(self,soln):

		"""
		Stroud/Xiu Sparse Samplers

		Compute integration points based on Stroud-2, Stroud-3, Xiu-2, or Xiu-3 cubature formulas of corresponding degrees:

		N : integer - number of random vectors to compute grids
		soln : integer - Which formula to use:

			(2) : Stroud deg-2 - uniform priors
			(3) : Stroud deg-3 - uniform priors
			(4) : Xiu deg-2 - Gaussian priors
			(5) : Xiu deg-3 - Gaussian priors

		Stroud Formulas are exact for symmetric uniform distribution pdfs
		For Stroud Formulas -See Stroud, 1957, "Remarks on the Disposition of
		Points in Numerical Integration Formula", MTAC, v.11, p. 257.

		Xiu Formulas are exact for symmetric Gaussian pdfs
		For Xiu Formulas - See Xiu, 2007, "Numerical Integration Formulas of
		Degree 2"
		"""

		N = self.d

		if soln == 2 : #Use Stroud degree-2 cubature nodes for Uniform Priors

			Y = npy.ndarray(shape=(N+1,N),dtype=float)

			if math.floor(N/2) == N/2.0:
				for k in range(0,N+1):
					for r in range(1,(N/2)+1):
						i = r
						Y[k,2*i-2] = npy.sqrt(2./3.)*(math.cos((2*r*k*math.pi)/(N+1)))
						Y[k,2*i-1] =   npy.sqrt(2./3.)*(math.sin((2*r*k*math.pi)/(N+1)))

			else:
					for k in range(0,N+1):
						for r in range(1,((N-1)/2)+1):
								i = r
								Y[k,2*i-2] = npy.sqrt(2./3.)*(math.cos((2*r*k*math.pi)/(N+1)))
								Y[k,2*i-1] =   npy.sqrt(2./3.)*(math.sin((2*r*k*math.pi)/(N+1)))
						#Employ odd formula for last node point
						Y[k,N-1] = ((-1)**k)/npy.sqrt(3.)

		elif soln == 3 : #Use Stroud degree-3 cubature nodes for Uniform Priors

			Y = npy.ndarray(shape=(N*2,N),dtype=float)

			if math.floor(N/2) == N/2.0:
					for k in range(0,2*N+1):
						for r in range(1,(N/2)+1):
								i = r
								Y[k-1,2*i-2] = npy.sqrt(2./3.)*(math.cos(((2*r-1)*k*math.pi)/(N)))
								Y[k-1,2*i-1] =   npy.sqrt(2./3.)*(math.sin(((2*r-1)*k*math.pi)/(N)))

			else:
					for k in range(0,2*N+1):
						for r in range(1,((N-1)/2)+1):
								i = r
								Y[k-1,2*i-2] = npy.sqrt(2./3.)*(math.cos(((2*r-1)*k*math.pi)/(N)))
								Y[k-1,2*i-1] =   npy.sqrt(2./3.)*(math.sin(((2*r-1)*k*math.pi)/(N)))
						#Employ odd formula for last node point
						Y[k-1,N-1] = ((-1)**k)/npy.sqrt(3.)

		elif soln == 4: #Use Xiu-2 cubature nodes for Gaussian Priors

			Y = npy.ndarray(shape=(N+1,N),dtype=float)

			if math.floor(N/2) == N/2.0:
					for k in range(0,N+1):
						for r in range(1,(N/2)+1):
								i = r
								Y[k,2*i-2] = npy.sqrt(2.)*(math.cos((2*r*k*math.pi)/(N+1)))
								Y[k,2*i-1] =   npy.sqrt(2.)*(math.sin((2*r*k*math.pi)/(N+1)))

			else:
					for k in range(0,N+1):
						for r in range(1,((N-1)/2)+1):
								i = r
								Y[k,2*i-2] = npy.sqrt(2.)*(math.cos((2*r*k*math.pi)/(N+1)))
								Y[k,2*i-1] =   npy.sqrt(2.)*(math.sin((2*r*k*math.pi)/(N+1)))
						#Employ odd formula for last node point
						Y[k,N-1] = ((-1)**k)

		elif soln == 5: #Use Xiu degree-3 cubature nodes for Gaussian Priors

			Y = npy.ndarray(shape=(N*2,N),dtype=float)

			if math.floor(N/2) == N/2.0:
					for k in range(0,2*N+1):
						for r in range(1,(N/2)+1):
								i = r
								Y[k-1,2*i-2] = npy.sqrt(2.)*(math.cos(((2*r-1)*k*math.pi)/(N)))
								Y[k-1,2*i-1] =   npy.sqrt(2.)*(math.sin(((2*r-1)*k*math.pi)/(N)))

			else:
					for k in range(0,2*N+1):
						for r in range(1,((N-1)/2)+1):
								i = r
								Y[k-1,2*i-2] = npy.sqrt(2.)*(math.cos(((2*r-1)*k*math.pi)/(N)))
								Y[k-1,2*i-1] =   npy.sqrt(2.)*(math.sin(((2*r-1)*k*math.pi)/(N)))
						#Employ odd formula for last node point
						Y[k-1,N-1] = ((-1)**k)

		else:
			raise Exception("Solution type must be 2-5")

		return Y

	def getseq(self):

		"""
		Helper method for cheby()

		Get the multi-indices sequence for sparse grids without computing
		full tensor products
		"""

		# Set local sampling parameters
		n = self.n
		d = self.d

		nl=it.combinations(range(n+d-1),d-1)
		nlevels = 0

		for i in nl:
			nlevels += 1

		seq = npy.zeros(shape=(nlevels,d),dtype=int)

		seq[0,0] = n
		maxi = n

		for k in range(int(1),nlevels):
			if seq[k-1,0] > int(0):
				seq[k,0] = seq[k-1,0] - 1
				for l in range(int(1),d):
					if seq[k-1,l] < maxi:
						seq[k,l] = seq[k-1,l] + 1
						for m in range(l+1,d):
							seq[k,m] = seq[k-1,m]
						break
			else:
				sum1 = int(0)
				for l in range(int(1),d):
					if seq[k-1,l] < maxi:
						seq[k,l] = seq[k-1,l] + 1
						sum1 = sum1 + seq[k,l]
						for m in range(l+1,d):
							seq[k,m] = seq[k-1,m]
							sum1 = sum1 + seq[k,m]
						break
					else:
						temp = int(0)
						for m in range(l+2,d):
							temp = temp + seq[k-1,m]
						maxi = n - temp
						seq[k,l] = 0
				seq[k,0] = n - sum1
				maxi = n - sum1

		return seq

if __name__ == "__main__":

	"""
	See /tests/grid_random_sampler_tests.py for 2D usage and unit tests for this class
	"""