"""
An interpolant function allows the evaluation of the fields for a given set of particles.
"""

from scipy.spatial import distance, KDTree
from astropy import units
import numpy as np

def isIterable(obj):
	"""
	Check if obj is iterable
	
	Parameters:
	obj: 
		object to check if iterable
	returns:
		bool
		True if obj is iterable, False otherwise
	"""
	try:
		iter(obj)
		return True
	except TypeError:
		return False

def evalKernel(xEval, xParticle, h, kernel):
	"""
	Helper function to evaluate kernel

	Parameters
	----------
	xEval:
		positions at which to evaluate kernel
	xParticle:
		positions of particles for which to evaluate kernel
	h:
		particle smoothing lengths
	kernel:
		kernel function

	Returns
	-------
		evaluation of kernel at xEval for particles at positions xParticle and smoothing lengths h
	"""
	q = distance.cdist(xEval/h, xParticle/h)
	return kernel(q) / (h ** 3)

def SPH(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, trigger = None):
	"""
	Compute the interpolated radial velocity, density and temperature fields using SPH interpolation evaluated at fieldPos positions
	Note that different SPH schemes have different definitions for velocity interpolation.
	This interpolant assumes that the conserved quantities are interpolated.

	Parameters
	----------
	X :
		particle positions
	V :
		particle radial velocities
	H : 
		particle smoothing lengths
	MHI :
		particle HI mass
	T :
		particle temperature in V**2 units
	M :
		particle mass
	kernel :
		kernel used in simulation
	fieldPos :
		positions at which to interpolate fields.
	dVolume :
		volume element size.
	trigger :
		unused

	Returns
	-------
	finalV : array astropy quantity
		interpolated velocity
	fieldMHI : array astropy quantity
		interpolated HI mass
	final T : array atropy quantity
		interpolated thermal velocity dispersion
	"""
	M *= units.dimensionless_unscaled
	N, nDim = X.shape
	if(V.ndim != 1):
		V = V[:,0] #more than one dimension of velocity is given, use radial velocity
	nPos = len(fieldPos)
	if not isIterable(dVolume):
		dVolume = np.ones(nPos) * dVolume
	slices = KDTree(fieldPos).query_ball_point(X, H)
	particleKernels = []
	fieldMHI = np.zeros(nPos) * MHI.unit / dVolume.unit
	fieldM = np.zeros(nPos) * M.unit / dVolume.unit
	fieldV = np.zeros(nPos) * V.unit * M.unit / dVolume.unit
	fieldT = np.zeros(nPos) * V.unit ** 2 * M.unit / dVolume.unit
	for i in range(N):
		particleKernel = evalKernel(fieldPos[slices[i]], X[i].reshape((1, nDim)), H[i], kernel)[:,0]
		fieldM[slices[i]] += particleKernel * M[i]
		fieldMHI[slices[i]] += particleKernel * MHI[i]
		fieldV[slices[i]] += particleKernel * V[i] * M[i] #quantity of movement is conserved
		fieldT[slices[i]] += particleKernel * T[i] * M[i] #thermal energy is conserved
	del slices
	kernelSlice = fieldM != 0
	finalV = np.zeros(nPos) * V.unit
	finalT = np.zeros(nPos) * V.unit ** 2
	finalV[kernelSlice] = fieldV[kernelSlice] / fieldM[kernelSlice]
	finalT[kernelSlice] = fieldT[kernelSlice] / fieldM[kernelSlice]
	return finalV, fieldMHI, finalT

def MFM(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, trigger = 1e4):
	"""
	Compute the interpolated radial velocity, density and temperature fields using MFM interpolation evaluated at flatPos positions
	If high mass particles are near the borders of the interpolation region, this will cause noticeable errors.

	Parameters
	----------
	X :
		particle positions
	V :
		particle radial velocities
	H : 
		particle smoothing lengths
	MHI :
		particle HI mass
	T :
		particle temperature in V**2 units
	M :
		particle mass
	kernel :
		kernel used in simulation
	fieldPos :
		positions at which to interpolate fields.
	dVolume :
		volume element size.
	trigger : int
		number of cells intersecting particle after which the memory isn't saved and kernel is recomputed.

	Returns
	-------
	finalV : array astropy quantity
		interpolated velocity
	finalP : array astropy quantity
		interpolated HI mass
	finalT : array atropy quantity
		interpolated thermal velocity dispersion
	"""
	M *= units.dimensionless_unscaled
	N, nDim = X.shape
	if(V.ndim != 1):
		V = V[:,0] #more than one dimension of velocity is given, use radial velocity
	nPos = len(fieldPos)
	if not isIterable(dVolume):
		dVolume = np.ones(nPos) * dVolume
	slices = KDTree(fieldPos).query_ball_point(X, H)
	totalKernel = np.zeros(nPos) / dVolume.unit
	particleKernels = []
	for i in range(N):
		particleKernel = kernel(fieldPos[slices[i]], X[i].reshape((1, nDim)), H[i])[:,0]
		totalKernel[slices[i]] += particleKernel
		if (len(slices[i]) > trigger):
			particleKernels += [True]
		else:
			particleKernels += [particleKernel]
	fieldMHI = np.zeros(nPos) * MHI.unit / (dVolume.unit ** 2)
	fieldM = np.zeros(nPos) * M.unit / (dVolume.unit ** 2)
	fieldV = np.zeros(nPos) * V.unit * M.unit / (dVolume.unit ** 2)
	fieldT = np.zeros(nPos) * V.unit ** 2 * M.unit / (dVolume.unit ** 2)
	for i in range(N):
		if len(slices[i]) == 0:
			continue
		currentKernel = particleKernels[i]
		if currentKernel is True:
			currentKernel = kernel(fieldPos[slices[i]], X[i].reshape((1, nDim)), H[i])[:,0]
		volume = np.sum( currentKernel * (dVolume[slices[i]] / totalKernel[slices[i]]) )
		#volume *=  np.pi*4/3 * H[i]**3 / np.sum(dVolume[slices[i]]) # for out of bounds particles, the volume is scaled up
		fieldMHI[slices[i]] += currentKernel * MHI[i] / volume
		fieldM[slices[i]] += currentKernel * M[i] / volume
		fieldV[slices[i]] += currentKernel * V[i] * M[i] / volume
		fieldT[slices[i]] += currentKernel * T[i] * M[i] / volume
	del slices, particleKernels #, fieldPos
	kernelSlice = totalKernel != 0
	finalV = np.zeros(nPos) * V.unit
	finalT = np.zeros(nPos) * V.unit ** 2
	finalP = np.zeros(nPos) * MHI.unit / dVolume.unit
	finalM = np.zeros(nPos) * M.unit / dVolume.unit
	finalP[kernelSlice] = fieldMHI[kernelSlice] / totalKernel[kernelSlice]
	finalM[kernelSlice] = fieldM[kernelSlice] / totalKernel[kernelSlice]
	finalV[kernelSlice] = fieldV[kernelSlice] / totalKernel[kernelSlice] / finalM[kernelSlice]
	finalT[kernelSlice] = fieldT[kernelSlice] / totalKernel[kernelSlice] / finalM[kernelSlice]
	return finalV, finalP, finalT

def voronoiMesh(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, trigger = 1e4):
	"""
	Compute the interpolated radial velocity, density and temperature fields using voronoi mesh.
	If high mass particles are near the borders of the interpolation region, this will cause noticeable errors.

	Parameters
	----------
	X :
		particle positions
	V :
		particle radial velocities
	H : 
		Unuseed.
		(particle volume)**(-3)
		Only used to convert MHI into a particle density.
		Exact volume is not required if you know the density. 
	MHI :
		particle HI mass
	T :
		particle temperature in V**2 units
	M :
		particle mass (unused)
	kernel :
		unused
	fieldPos :
		positions at which to interpolate fields.
	dVolume :
		volume element size for fieldPos
	trigger : int
		unused

	Returns
	-------
	fieldV : array astropy quantity
		interpolated velocity
	fieldMHI : array astropy quantity
		interpolated HI mass
	fieldT : array atropy quantity
		interpolated thermal velocity dispersion
	"""
	M *= units.dimensionless_unscaled
	N, nDim = X.shape
	if(V.ndim != 1):
		V = V[:,0] #more than one dimension of velocity is given, use radial velocity
	particleSlices = KDTree(X).query(fieldPos)[1] #nearest neighbor assignment of particles to field pos
	particleMasks = particleSlices == np.arange(len(X))[:, np.newaxis]
	particleVolumes = np.sum(particleMasks * dVolume, axis = 1)
	del particleMasks
	fieldMHI = MHI[particleSlices] / particleVolumes[particleSlices]
	fieldV = V[particleSlices]
	fieldT = T[particleSlices]
	return fieldV, fieldMHI, fieldT
