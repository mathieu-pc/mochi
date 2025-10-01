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

def SPH(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, **kwargs):
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

def MFM(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, kernelRecalculationTrigger = 1e4, **kwargs):
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
	kernelRecalculationTrigger : int
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
		particleKernel = evalKernel(fieldPos[slices[i]], X[i].reshape((1, nDim)), H[i], kernel)[:,0]
		totalKernel[slices[i]] += particleKernel
		if (len(slices[i]) > kernelRecalculationTrigger):
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
			currentKernel = evalKernel(fieldPos[slices[i]], X[i].reshape((1, nDim)), H[i], kernel)[:,0]
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

def voronoiMesh(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, **kwargs):
	"""
	Compute the interpolated radial velocity, density and temperature fields using voronoi mesh.
	Assumes that fieldPos creates a box.

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
	particleIndices = np.arange(len(X))
	_, nearestParticleIndices = KDTree(X).query(fieldPos) #nearest neighbor assignment of particles to field pos

	#construct a mask for inbound particles but not assigned to a cell
	inboundParticleMask = np.all(X > fieldPos.min(axis = 0), axis = 1) & np.all(X < fieldPos.max(axis = 0), axis = 1) #assume box shape for field pos
	usedParticleMask = np.isin(particleIndices, nearestParticleIndices)
	missedParticleMask = inboundParticleMask & ~usedParticleMask
	missedParticleIndices = particleIndices[missedParticleMask]
	_, missedParticleCellIndices = KDTree(fieldPos).query(X[missedParticleMask])
	nMissedParticle = np.sum(missedParticleMask)

	particleMasks = nearestParticleIndices == particleIndices[:, np.newaxis]
	particleMasks[missedParticleIndices, missedParticleCellIndices] = True

	fieldNParticle = np.ones(len(fieldPos), dtype = int)
	fieldNParticle[missedParticleCellIndices] += 1

	particleVolumes = np.einsum('ij,j->i', particleMasks, dVolume / fieldNParticle) #for shared cells, the volume is divided between the particles
	#particleVolumes = np.sum(particleMasks * dVolume / fieldNParticle, axis = 1) #for shared cells, the volume is divided between the particles

	fieldMHI = MHI[nearestParticleIndices] / particleVolumes[nearestParticleIndices]
	fieldMHI[missedParticleCellIndices] += MHI[missedParticleMask] / particleVolumes[missedParticleMask]
	fieldMHI /= fieldNParticle

	fieldV = V[nearestParticleIndices]
	fieldV[missedParticleCellIndices] += V[missedParticleMask]
	fieldV /= fieldNParticle

	fieldT = T[nearestParticleIndices]
	fieldT[missedParticleCellIndices] += T[missedParticleMask]
	fieldT /= fieldNParticle

	return fieldV, fieldMHI, fieldT
