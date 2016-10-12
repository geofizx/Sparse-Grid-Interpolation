#!/usr/bin/env python2.7
# encoding: utf-8

"""
Perform n-d sparse grid interpolation (See sparse_vals.py for variable description)
using Clenshaw-Curtis or Chebyshev polynomials

NOTE:
#z_k are surpluses are defined as difference between function (fun_nd) evaluations at current sparse
#grid nodes [grdin] and interpolated values (runInterp) on the same grid using previous
#grid level interpolant. It measures the error between the current function
#values and those estimated with the previous grid  level interpolation.
#e.g., zk(@ k=2) = fun_nd(grdin, @k=2) - runInterp(grdin,@k=1)

@author Michael Tompkins
@copyright 2016
"""

import numpy as npy

def runInterp(d,z_k,grdin,grdout,indx,mi,type0,intvl):

	"""
	Perform n-d sparse grid interpolation (See sparse_vals.py for variable description)
	:arg d : int dimensionality of interpolation domain
	:arg z_k : array of hierarchical surpluses as defined in sparse_vals.py
	:arg grdin : array of input nodes for interpolation
	:arg grdout : array of output points for interpolation
	:arg mi : list of number of nodal points used in each 1-D basis of the mult-dimensional interpolation
	:arg type0 : string for polynomial type "CC" - Clenshaw Curtis, or "CH" - Chebyshev
	:arg indx : array of multi-indexes as described in sparse_vals.py
	:return ip2 : array of interpolated values at points specified in array grdout
	"""

	num2 = grdout.shape[0]
	num4 = indx.shape[0]
	ipmj = npy.zeros(shape=(num2),dtype=float)
	ip2 = npy.zeros(shape=(num2),dtype=float)

	wght = npy.zeros(shape=(num4,d),dtype=float) # Initialize weights for current level k
	wght2 = npy.ones(shape=(num4),dtype=float)
	polyw = npy.zeros(shape=(num4,d),dtype=float)

	# Now loop over indices and grid points to perform interpolate at
	# current grid level, k, using surpluses, zk.
	#
	# Formulas based on piecewise multi-linear basis functions of the kind:
	# wght2_j = 1-(mi-1)*norm(x-x_j), if norm(x-x_j)> 1/(mi-1), else wght2_j = 0.0
	if type0 == 'cc' or type0 == 'CC':
		for i in range(0,num2):					# Number of points to interpolate
			for j in range(0,num4):				# Number of grid nodes at current level
				wght2[j] = 1.0					# Iinitialize total linear basis integration weights
				for l in range(0,d): 			# Number of dimensions for the interpolation
					# Determine 1D linear basis functions for each index i
					if mi[indx[j,l]] == 1:
						wght[j,l] = 1.0			# Leave weight == 1.0 if mi = 1
					elif npy.linalg.norm(grdout[i,l]-grdin[j,l]) < (1./(mi[indx[j,l]]-1)):
						wght[j,l] = 1-(mi[indx[j,l]]-1)*npy.linalg.norm(grdout[i,l]-grdin[j,l])   # Compute 1D linear basis functions
					else:
						wght[j,l] = 0.0			# Compute 1D linear basis functions
					wght2[j] *= wght[j,l]		# Perform the dimensional products for the basis functions
				ipmj[i] += wght2[j]*z_k[j]		# Sum over the number of total node points (j=num4) for all dimensions
			ip2[i] = ipmj[i] 					# Re-assign the interpolated value to new variable (redundant)

	# Formulas based on Barycentric Chebyshev polynomial basis functions of the kind:
	# wght2_j = SUM_x_m[(x - x_m)/(x_j - x_m)], for all x_m != x_j
	elif type0 == 'ch' or type0 == 'CH':
		for i in range(0,num2):					# Number of points to interpolate
			for j in range(0,num4):				# Number of grid nodes at current level
				wght2[j] = 1.0					# Initialize total Chebyshev integration weights (i.e., w(x))
				for l in range(0,d): 			# Number of dimensions for the interpolation
					polyw[j,l] = 1.0           	# Iinitialize d-dim polynomial (i.e., (x - x_m)/(x_m - x_j))
					if mi[indx[j,l]] != 1:		# Leave weight == 1.0 if mi = 1
						for m in range(0,mi[indx[j,l]]):		#m=1:mi(indx(j,l)):   # Else compute weight products over number of nodes for a given mi
							xtmp = (1.+(-npy.cos((npy.pi*m)/(mi[indx[j,l]]-1))))/2.    # Compute 1D node position on-the-fly
							# Transform xtmp based on interval
							range1 = npy.abs(npy.min(intvl[:,l])-npy.max(intvl[:,l]) )
							xtmp = xtmp*range1 + min(intvl[:,l])
							if npy.abs(grdin[j,l] - xtmp) > 1.0e-03:	# Polynomial not defined if xtmp==grdin(j,l)
								polyw[j,l] *= (grdout[i,l]-xtmp)/(grdin[j,l]-xtmp)  # Perform 1D polynomial products
					wght2[j] *= polyw[j,l]         # Perform the dimensional products for the polynomials
				ipmj[i] += wght2[j]*z_k[j]			# Sum over the number of total node points (j=num4) for all dimensions
			ip2[i] = ipmj[i]                               # Now re-assign the interpolated value to new variable (redundant)
	else:
		raise Exception('error: type must be "cc" or "ch"')

	#ip2 = ip2.T                   # Take the transpose of the interpolated vector output to conform to other routine outputs

	return ip2
