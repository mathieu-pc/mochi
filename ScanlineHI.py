import numpy as np
from scipy.spatial import KDTree
from astropy import units, constants
from scipy.spatial import distance

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

def makeGrid(shape, deltaX, ndim = 3):
	"""
	Make a grid to interpolate fields on

	Parameters
	----------
	shape: tuple (int)
		shape of grid
	deltaX:
		step of the grid
	ndim: int
		number of dimensions
	
	Returns
	-------
	: array prod(shape) by ndim
		positions of grid
	"""
	if not isIterable(shape):
		shape = (shape,) * ndim
	if not isIterable(deltaX):
		deltaX = np.ones(ndim) * deltaX
	return _makeGrid(shape, deltaX)

def _makeGrid(shape, deltaX):
	"""
	Make a grid to interpolate fields on

	Parameters
	----------
	shape: tuple (int)
		shape of grid
	deltaX:
		step of the grid
	
	Returns
	-------
	: array prod(shape) by len(shape)
		positions of grid
	"""
	coordinateRanges = [ (np.arange(shape[i])- (shape[i]-1)/2) * deltaX[i] for i in range(len(shape))]
	coordinates = np.meshgrid(*coordinateRanges, indexing = 'ij')
	return np.stack([line.flatten() for line in coordinates], axis = -1)

def getChannelNumber(VX, M, T, dV, *, nMin = 120, nMax = 300):
	"""
	Utility function
	Estimates the best number of channels to get most of the cube flux in.

	Parameters
	----------
	VX : 
		array of interpolated radial velocities
	M : 
		array of interpolated masses
	T :
		array of interpolated temperatures
	dV :
		volume element
	nMin :
		minimum number of channels
	nMax :
		maximum number of channels

	Returns
	-------
		(int)
		channel number to capture HI
	"""
	sorter = np.argsort(VX)
	v = VX[sorter]
	m = M[sorter]
	t = T[sorter]
	mIntegrate = np.cumsum(m)
	mIntegrate /= mIntegrate[-1]
	i1 = np.searchsorted(mIntegrate, 0.975)
	i2 = np.searchsorted(mIntegrate, 0.025)
	if v[i1] > np.abs(v[i2]):
		i = i1
	else:
		i = i2
	guess = int((np.abs(v[i]) + 3 * np.sqrt(t[i]))/dV + 1 )
	return max(min((guess + 25)*2, nMax), nMin)+1

def interpolateSPH(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, trigger = None):
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

def interpolateMFM(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, trigger = 1e4):
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

def makeCubeFromFields(fieldMHI, fieldV, fieldT, channelSize, dVolume, shape):
	"""
	Assemble fields into an HI cube
	
	Parameters
	----------
	fieldMHI:
		HI masses
	fieldV:
		radial velocities
	fieldT:
		velocity dispersions
	channelSize:
		size of channel in velocity units
	dVolume:
		volume elements
	shape:
		spatial shape of fieldMHI, fieldV, fieldT

	Returns
	-------
	mock cube
	"""
	nChannel = getChannelNumber(fieldV, fieldMHI, fieldT, channelSize)
	spectrumRange = (channelSize * (np.arange(nChannel) - (nChannel-1)/2)).reshape(nChannel, 1, 1, 1)
	fieldMHI = fieldMHI.reshape(shape)
	fieldT = fieldT.reshape(shape)
	fieldV = fieldV.reshape(shape)
	fieldT[fieldMHI==0] = 1 * fieldT.unit
	numerator = fieldMHI / np.sqrt(2*np.pi*fieldT) * channelSize * dVolume
	del fieldMHI
	cube = np.zeros( (nChannel, shape[1], shape[2]) ) * numerator.unit
	for i in range(nChannel):
		cube[i] += np.sum( numerator * np.exp(-(fieldV-spectrumRange[i])**2/fieldT/2), axis = 0)
	return np.flip(np.moveaxis(cube, 1, 2), axis = 2)

def makeCube(shape, deltaX, particles, kernel, channelSize, interpolant, *, trigger = 1e4):
	"""
	Make a cube using scanline method.
	Parameters
	----------
	shape : tuple (int)
		number of elements across each dimension
	deltaX :
		Physical pixel size.
		Recommend to use half the minimum smoothing length.
		Above that, the spectrum changes with the chosen size.
		The cube can be downsampled using openCV2.
	particles :
		particles in dict format like MARTINI
	kernel :
		kernel function
	channel size :
		spectral channel size
	interpolant :
		interpolation method (ex: SPH)
	
	Returns
	-------
		mock HI cube
	"""
	if not isIterable(shape):
		shape = (shape,) * 3
	if not isIterable(deltaX):
		deltaX = np.ones(3) * deltaX
	dVolume = np.prod(deltaX.value) * (deltaX.unit ** 3)
	fieldV, fieldMHI, fieldT = interpolant(
		particles["xyz_g"],
		particles["vxyz_g"],
		particles["hsm_g"],
		particles["mHI_g"],
		particles["T_g"],
		particles["m"],
		kernel,
		makeGrid(shape, deltaX),
		dVolume,
		trigger = trigger
	)
	return makeCubeFromFields(fieldMHI, fieldV, fieldT, channelSize, dVolume, shape)
