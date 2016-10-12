#!/usr/bin/env python2.7
# encoding: utf-8

"""
@desciption

Class to perform hierarchical sparse-grid polynomial interpolation at multiple grid levels using
either piece-wise linear Clenshaw-Curtis (type0 = 'CC') or Chebyshev polynomial (type0 = 'CH') basis functions at
sparse grid nodes specified by max degree (i.e., level) of interpolation and dimensionality of space.

Early stopping is implemented when absolute error at any level is less than tol

@usage
[ip2,wk,werr,meanerr] = sparse_vals(maxn,d,type0,intvl,fun_nd,grdout)

	:arg

	maxn : integer : maxlevel of interpolation depth
	d : integer : dimension of interpolation
	type0 : string : polynomial : piece-wise linear (type0 = 'CC') or Chebyshev polynomial (type0 = 'CH') to use for interpolation
	intvl : 2 x d array : interval other than [0 1] for each dimension over which to compute sparse grids
	fun_nd : function : user-defined function used to evaluate the target function at sparse grid nodes
	grdout : N x d array : desired N points in d-dimensions to interpolate to

	:return
	ip2 : N-length array of interpolated values on the user specified grid grdout
	werr : absolute error of interpolation for each level [0-maxn]
	meanerr : mean error of interpolation for each level [0-maxn]
	wk : hiearchical surpluses for each interpolation level [0-maxn]

@dependencies
samplers.py - Companion script for computing sparse grid node points
spinterp.py - Companion script for computing interpolation at each hierarchical level

@references
See Klemke, A. and B. Wohlmuth, 2005, Algorithm 847: spinterp: Piecewise
Multilinear Hierarchical Sparse Grid Interpolation in MATLAB,
ACM Trans. Math Soft., 561-579.

@author Michael Tompkins
@copyright 2016
"""

# Externals
import numpy as npy

# Internals
from code import fun_nd,spinterp,samplers


class sparseInterp():

	def __init__(self, maxn, dimensions, grdout, type0, intvl=None):

		"""
		:arg maxn : integer maximum degree to consider for hierarchical sparse-grid interpolation
		:arg dimensions : integer dimensionality of sampling space
		:arg grdout : N x d array : User specified N output grid points to interpolate function to
		:arg type0 : string specifies base polynomial for interpolation (Chebyshev = "CH", Clenshaw-Curtis - "CC")
		:return : ip2 : array-type (N,d) of interpolated function values

		options - debug : turn on (True) or off (False) print statements
		"""

		self.grdout = grdout			# User specified output grid points to interpolate function to
		self.maxn = maxn				# Maximum degree of interpolation to perform -- see self.tol for early stopping
		self.d = dimensions				# Number of dimensions of interpolation
		self.dim1 = grdout.shape[0]		# Number of samples for output
		self.dim2 = grdout.shape[1]		# Dimensionality of interpolation
		self.debug = False				# Include print statements to stdout
		self.debug = 0					# 0 = user defined function used, 1 = unit test fun2d used
		self.tol = 0.001				# Early stopping criteria
		self.type = type0				# Type of polynomial to perform interpolation
		self.intvl = intvl				# Interval over which to perform interpolation

	def runInterp(self):

		"""
		Perform n-d sparse grid interpolation
		"""

		grdout = self.grdout
		intvl = self.intvl
		maxn = self.maxn
		d = self.d
		num2 = self.dim1								# Number of points to interpolate on user input grid
		ip2 = npy.zeros(shape=num2,dtype=float)			# Initialize final interpolated array
		ipmj = npy.zeros(shape=num2,dtype=float)		# Initialize d-variate interpolant array
		tol = self.tol									# Early stopping criteria for interpolation

		grdbck = {}		# Dictionary for back storage of grid arrays at each level k, for hierarchical error checking
		indxbck = {}	# Dictionary for back storage of grid index arrays at each level k, for hierarchical error checking
		mibck = {}		# Dictionary for back storage of grid sample # at each level k, for hierarchical error checking
		yk = {}			# Dictionary of function evaluations for each level k
		wk = {}			# Dictionary of hierarchical surpluses for each level k
		meanerr = {}	# Dictionary of hierarchical surpluses mean errors for each level k
		werr = {}		# Dictionary of hierarchical surpluses absolute errors for each level k

		# Loop over all grid levels (i.e., polynomial degree) from k=0:maxn to determine optimal level for interpolation
		# Break criteria is when mx{zk} <= toler
		for k in xrange(0,maxn+1):

			"""
			Determine index sets and sparse grid nodes for each grid level k and dimension d, type of interpolation,
			and interval [0 1]
			"""
			samp = samplers.ndSampler(k,d)					# Instantitate sampler class for current level k (degree) of interpolation
			grdin,mi,indx = samp.sparse_sample(self.type)		# Compute polynomial nodes for each level k of interpolation

			if k == 0:
				indx = indx[npy.newaxis,:]		# Add dimension when 1-D array returned for level 0 grid

			# Stretch/squeeze grid to interval [intvl] in each dimension
			for i in xrange(0,d):
				range1 = abs( min(intvl[:,i]) - max(intvl[:,i]) )
				grdin[:,i] = grdin[:,i]*range1 + min(intvl[:,i])

			grdbck[k] = grdin          					# Back storage of grid k
			indxbck[k] = indx							# Back storage of multi-index array at level k
			mibck[k] = mi								# Back storage of node number array
			num4 = indx.shape[0]						# Number of multi-indices to compute cartesian products
			wght = npy.zeros(shape=(num4,d),dtype=float)	# Initialize weights for CC current level k
			wght2 = npy.ones(shape=(num4),dtype=float)		# Interpolation weights
			polyw = npy.zeros(shape=(num4,d),dtype=float)	# Initialize weights for CH current level k

			# Determine function values at current kth sparse grid nodes using user-defined function fun_nd
			try:
				yk[k] = fun_nd.fun_nd(grdin)
			except:
				raise Exception("User-defined function missing or behaving abnormally")

			# Initialize surpluses to current grid node values
			zk = yk[k]

			# Compute hierarchical surpluses by subtracting interpolated values of current grid nodes runInterp(grdin)
			# computed at grid level k-1 interpolant from current function values (fun_nd(x)) computed at current grid
			# level, k, e.g., zk(@ k=2) = fun_nd(grdin, @k=2) - runInterp(grdin, @k=1)

			# This allows for the determination of error at each grid level and a simpler implementation
			# of the muti-variate interpolation at various Smoyak grid levels.
			if k > 0:
				if werr[k-1] < tol:       # Stop criteria based on average surplus error
					if self.debug is True:
						print 'Mean error tolerance met at Grid Level...',str(k-1)
					return ip2,meanerr,werr
				else:
					for m in range(0,k): #i=0:k-1          # Must loop over all levels to get complete interpolation (@ k-1)
						runterp = spinterp.runInterp(d,wk[m],grdbck[m],grdin,indxbck[m],mibck[m],self.type,intvl)
						zk -= runterp

			# Loop over indices and grid points to perform interpolate at current grid level, k, using surpluses, zk.

			# Formulas based on Clenshaw-Curtis piecewise multi-linear basis functions of the kind:
			# wght2_j = 1-(mi-1)*norm(x-x_j), if norm(x-x_j)> 1/(mi-1), else wght2_j = 0.0
			if self.type == 'cc' or self.type == 'CC':
				for i in range(0,num2):					# Number of points to interpolate
					for j in range(0,num4):				# Number of grid nodes at current level
						wght2[j] = 1.0					# Iinitialize total linear basis integration weights
						for l in range(0,d): 			# Number of dimensions for the interpolation
							# Determine 1D linear basis functions for each index i
							if mi[indx[j,l]] == 1:
								wght[j,l] = 1.0	# Leave weight == 1.0 if mi = 1
							elif npy.linalg.norm(grdout[i,l]-grdin[j,l]) < (1./(mi[indx[j,l]]-1)):
								wght[j,l] = 1-(mi[indx[j,l]]-1)*npy.linalg.norm(grdout[i,l]-grdin[j,l])   # Compute 1D linear basis functions
							else:
								wght[j,l] = 0.0			# Compute 1D linear basis functions
							wght2[j] *= wght[j,l]		# Perform the dimensional products for the basis functions
						ipmj[i] += wght2[j]*zk[j]		# Sum over the number of total node points (j=num4) for all dimensions
					ip2[i] = ipmj[i] 					# Now re-assign the interpolated value to new variable (redundant)

			# Formulas based on Barycentric Chebyshev polynomial basis functions of the kind:
			# wght2_j = SUM_x_m[(x - x_m)/(x_j - x_m)], for all x_m != x_j
			elif self.type == 'ch' or self.type == 'CH':
				for i in range(0,num2):					# Number of points to interpolate
					for j in range(0,num4):				# Number of grid nodes at current level
						wght2[j] = 1.0					# Initialize total Chebyshev integration weights (i.e., w(x))
						for l in range(0,d):			# Number of dimensions for the interpolation
							polyw[j,l] = 1.0			# Iinitialize d-dim polynomial (i.e., (x - x_m)/(x_m - x_j))
							if mi[indx[j,l]] != 1:		# Leave weight == 1.0 if mi = 1
								for m in range(0,mi[indx[j,l]]):		#m=1:mi(indx(j,l)):   # Else compute weight products over number of nodes for a given mi
									xtmp = (1.+(-npy.cos((npy.pi*(m))/(mi[indx[j,l]]-1))))/2.    # Compute 1D node position on-the-fly
									# Transform xtmp based on interval
									range1 = npy.abs(npy.min(intvl[:,l])-npy.max(intvl[:,l]) )
									xtmp = xtmp*range1 + npy.min(intvl[:,l])
									if npy.abs(grdin[j,l] - xtmp) > 1.0e-03:	# Polynomial not defined if xtmp==grdin(j,l)
										polyw[j,l] *= (grdout[i,l]-xtmp)/(grdin[j,l]-xtmp)	# Perform 1D polynomial products
							wght2[j] *= polyw[j,l]         # Perform the dimensional products for the polynomials
						ipmj[i] += wght2[j]*zk[j]			# Sum over the number of total node points (j=num4) for all dimensions
					ip2[i] = ipmj[i]                               # Now re-assign the interpolated value to new variable (redundant)

			else:
				raise Exception('error: type must be "cc" or "ch"')

			wk[k] = zk                       	   # Assign current surpluses to wk for back storage and output
			werr[k] = npy.max(npy.abs(wk[k]))      # Compute absolute error of current grid level
			meanerr[k] = npy.mean(npy.abs(wk[k]))  # Compute mean error of current grid level

		return ip2,meanerr,werr


if __name__ == "__main__":

	"""
	Unit tests - see /tests/sparse_grid_inter_tests.py for useage and unit tests for this class
	"""


