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

def makeCube(shape, deltaX, particles, kernel, channelSize, interpolant, radiativeTransferModel, **kwargs):
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
		**kwargs
	)
	return radiativeTransferModel(fieldMHI, fieldV, fieldT, channelSize, dVolume, shape)
