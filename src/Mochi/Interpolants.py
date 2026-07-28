"""
An interpolant function allows the evaluation of the fields for a given set of particles.
"""

from scipy.spatial import distance, KDTree
from sklearn.neighbors import KDTree as lKDTree
from astropy import units
import numpy as np
from functools import partial


def _isIterable(obj):
	"""
	Check if obj is iterable
	"""
	try:
		iter(obj)
		isIterable = True
	except TypeError:
		isIterable = False
	return isIterable


def _evalKernel(xEval, xParticle, h, kernel):
	"""
	Helper function to evaluate kernel
	"""
	q = distance.cdist(xEval/h, xParticle/h)
	return kernel(q) / (h ** 3)


def _evalCacheKernel(q, kernelCache, kernelCacheResolution):
	return kernelCache[(np.clip(q, 0, 1) * kernelCacheResolution).astype(np.uint8)]


def sphLoop(M, MHI, particlesMomentum, T, H, dist, slices, cellsVolume, kernelCache, kernelCacheResolution, nPos, nParticles, velocityUnit, massUnit, volumeUnit, maskOutOfBound):
	fieldMHI = np.zeros(nPos)
	fieldM = np.zeros(nPos)
	fieldV = np.zeros(nPos)
	fieldT = np.zeros(nPos)
	H3 = H ** 3
	for i in range(nParticles):
		if len(slices[i]) == 0:
			continue
		particleKernel = _evalCacheKernel(dist[i]/H[i], kernelCache, kernelCacheResolution) / H3[i]
		if not maskOutOfBound[i]: #Since the particle is not out bound, we know the kernel should sum to 1. The kernel not summing to 1 is due to resolution effects.
			particleKernel /= np.sum(particleKernel * cellsVolume[slices[i]])
		fieldM[slices[i]] += particleKernel * M[i]
		fieldMHI[slices[i]] += particleKernel * MHI[i]
		fieldV[slices[i]] += particleKernel * particlesMomentum[i]
		fieldT[slices[i]] += particleKernel * T[i]
	kernelSlice = fieldM != 0
	finalV = np.zeros(nPos) * velocityUnit
	finalT = np.zeros(nPos) * velocityUnit ** 2
	finalMHI = fieldMHI * massUnit / volumeUnit
	finalV[kernelSlice] = fieldV[kernelSlice] * velocityUnit / fieldM[kernelSlice]
	finalT[kernelSlice] = fieldT[kernelSlice] * velocityUnit ** 2 / fieldM[kernelSlice]
	fields = finalV, finalMHI, finalT
	return fields


def mfmLoop(M, MHI, particlesMomentum, T, H, dist, slices, cellsVolume, kernelCache, kernelCacheResolution, nPos, nParticles, velocityUnit, massUnit, volumeUnit, maskOutOfBound):
	fieldMHI = np.zeros(nPos)
	fieldM = np.zeros(nPos)
	fieldV = np.zeros(nPos)
	fieldT = np.zeros(nPos)
	H3 = H ** 3
	totalKernel = np.zeros(nPos)
	for i in range(nParticles):
		if len(slices[i]) == 0:
			continue
		particleKernel = _evalCacheKernel(dist[i]/H[i], kernelCache, kernelCacheResolution) / H3[i]
		totalKernel[slices[i]] += particleKernel
		slices[i] = slices[i][particleKernel != 0]
		dist[i] = dist[i][particleKernel != 0]
	fieldMHI = np.zeros(nPos)
	fieldM = np.zeros(nPos)
	fieldV = np.zeros(nPos)
	fieldT = np.zeros(nPos)
	for i in range(nParticles):
		if len(slices[i]) == 0:
			continue
		particleKernel = _evalCacheKernel(dist[i]/H[i], kernelCache, kernelCacheResolution) / H3[i]
		volume = np.sum( particleKernel * (cellsVolume[slices[i]] / totalKernel[slices[i]]) )
		if maskOutOfBound[i]:
			volume *=  np.pi*4/3 * H[i]**3 / np.sum(cellsVolume[slices[i]]) # for out of bounds particles, the volume is scaled up
		fieldMHI[slices[i]] += particleKernel * MHI[i] / volume
		fieldM[slices[i]] += particleKernel * M[i] / volume
		fieldV[slices[i]] += particleKernel * particlesMomentum[i] / volume
		fieldT[slices[i]] += particleKernel * T[i] / volume
	kernelSlice = totalKernel != 0
	finalV = np.zeros(nPos) * velocityUnit
	finalT = np.zeros(nPos) * velocityUnit ** 2
	finalMHI = np.zeros(nPos) * massUnit / volumeUnit
	finalM = np.zeros(nPos)
	finalMHI[kernelSlice] = fieldMHI[kernelSlice] * massUnit / volumeUnit / totalKernel[kernelSlice]
	finalM[kernelSlice] = fieldM[kernelSlice] / totalKernel[kernelSlice]
	finalV[kernelSlice] = fieldV[kernelSlice] * velocityUnit / totalKernel[kernelSlice] / finalM[kernelSlice]
	finalT[kernelSlice] = fieldT[kernelSlice] * velocityUnit ** 2 / totalKernel[kernelSlice] / finalM[kernelSlice]
	fields = finalV, finalMHI, finalT
	return fields


def _getOutOfBoundParticles(particlePos, particleRadius, fieldPos):
	lowBound = np.min(fieldPos, axis = 0)
	topBound = np.max(fieldPos, axis = 0)
	maskOutOfBound = ((particlePos + particleRadius[:,np.newaxis]) > topBound) | ((particlePos - particleRadius[:,np.newaxis]) < lowBound)
	maskOutOfBound = np.any(maskOutOfBound, axis = 1)
	return maskOutOfBound


def particleScatter(subInterpolant, X, V, H, MHI, T, M, kernel, fieldPos, dVolume, *, kernelCacheResolution = 256, **kwargs):
	kernelCache = kernel(np.linspace(0, 1, kernelCacheResolution))
	maskOutOfBound = _getOutOfBoundParticles(X, H, fieldPos)
	M *= units.dimensionless_unscaled
	N, nDim = X.shape
	if(V.ndim != 1):
		V = V[:,0] #more than one dimension of velocity is given, use radial velocity
	nPos = len(fieldPos)
	if not _isIterable(dVolume):
		dVolume = np.ones(nPos) * dVolume
	slices, dist = lKDTree(fieldPos.value).query_radius(X.value, H.value, return_distance = True)
	particleKernels = []
	particlesMomentum = V.value * M.value
	thermal = T.to_value(V.unit ** 2) * M.value
	fields = subInterpolant(
		M.value,
		MHI.value,
		particlesMomentum,
		thermal,
		H.value,
		dist,
		slices,
		dVolume.value,
		kernelCache,
		kernelCacheResolution,
		nPos,
		N,
		V.unit,
		MHI.unit,
		H.unit ** 3,
		maskOutOfBound
	)
	return fields


def composeParticleScatter(subInterpolant):
	def interpolant(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, *, kernelCacheResolution = 256, **kwargs):
		fields = particleScatter(subInterpolant, X, V, H, MHI, T, M, kernel, fieldPos, dVolume, kernelCacheResolution = 256, **kwargs)
		return fields
	return interpolant


SPH = composeParticleScatter(sphLoop)
MFM = composeParticleScatter(mfmLoop)


def _evalVoronoiField(particleQuantities, nearestParticleIndices, missedParticleCellIndices, missedParticleMask, fieldNParticle):
	fieldQuantity = particleQuantities[nearestParticleIndices]
	fieldQuantity[missedParticleCellIndices] += particleQuantities[missedParticleMask]
	fieldQuantity /= fieldNParticle
	return fieldQuantity


def voronoiMesh(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, **kwargs):
	"""
	Compute the interpolated radial velocity, density and temperature fields using voronoi mesh.
	Assumes that fieldPos creates a box.W
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

	fieldNParticle = np.ones(len(fieldPos), dtype = np.uint64)
	fieldNParticle[missedParticleCellIndices] += 1

	particleVolumes = np.einsum('ij,j->i', particleMasks, dVolume / fieldNParticle) #for shared cells, the volume is divided between the particles
	density = np.zeros(MHI.shape) * MHI.unit / particleVolumes.unit
	volumeMask = ~ (particleVolumes == 0)
	density[volumeMask] = MHI[volumeMask] / particleVolumes[volumeMask]
	fieldV =  _evalVoronoiField(V, nearestParticleIndices, missedParticleCellIndices, missedParticleMask, fieldNParticle)
	fieldMHI = _evalVoronoiField(density, nearestParticleIndices, missedParticleCellIndices, missedParticleMask, fieldNParticle)
	fieldT =  _evalVoronoiField(T, nearestParticleIndices, missedParticleCellIndices, missedParticleMask, fieldNParticle)
	fields = fieldV, fieldMHI, fieldT
	return fields


def manualSPH(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, **kwargs):
	"""
	Compute the interpolated radial velocity, density and temperature fields using SPH interpolation evaluated at fieldPos positions
	Note that different SPH schemes have different definitions for velocity interpolation.
	This interpolant assumes that the conserved quantities are interpolated.
	This SPH interpolant serves for testing purposes and writes the equations out explicitely.
	Consequently, it is slow but safe.
	"""
	M *= units.dimensionless_unscaled
	N, nDim = X.shape
	if(V.ndim != 1):
		V = V[:,0] #more than one dimension of velocity is given, use radial velocity
	nPos = len(fieldPos)
	if not _isIterable(dVolume):
		dVolume = np.ones(nPos) * dVolume
	slices = KDTree(fieldPos).query_ball_point(X, H)
	particleKernels = []
	fieldMHI = np.zeros(nPos) * MHI.unit / dVolume.unit
	fieldM = np.zeros(nPos) * M.unit / dVolume.unit
	fieldV = np.zeros(nPos) * V.unit * M.unit / dVolume.unit
	fieldT = np.zeros(nPos) * V.unit ** 2 * M.unit / dVolume.unit
	for i in range(N):
		particleKernel = _evalKernel(fieldPos[slices[i]], X[i].reshape((1, nDim)), H[i], kernel)[:,0]
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
	fields = finalV, fieldMHI, finalT
	return fields

