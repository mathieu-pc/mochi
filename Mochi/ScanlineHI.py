import numpy as np
from astropy import units

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
