import numpy as np
from astropy import units

def _isIterable(obj):
	"""
	Check if obj is iterable
	"""
	try:
		iter(obj)
		return True
	except TypeError:
		return False

def _makeGrid(shape, deltaX, ndim = 3):
	"""
	Make a grid to interpolate fields on
	"""
	if not _isIterable(shape):
		shape = (shape,) * ndim
	if not _isIterable(deltaX):
		deltaX = np.ones(ndim) * deltaX
	coordinateRanges = [ (np.arange(shape[i])- (shape[i]-1)/2) * deltaX[i] for i in range(len(shape))]
	coordinates = np.meshgrid(*coordinateRanges, indexing = 'ij')
	return np.stack([line.flatten() for line in coordinates], axis = -1)

def makeCube(shape, deltaX, particles, kernel, channelWidth, interpolant, radiativeTransferModel, **kwargs):
	"""
	Make a cube by 2 step process.
	1. Interpolate in simulation space using interpolant.
	2. Collapse using radiativeTrasnferModel.
	"""
	if not _isIterable(shape):
		shape = (shape,) * 3
	if not _isIterable(deltaX):
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
		_makeGrid(shape, deltaX),
		dVolume,
		**kwargs
	)
	return radiativeTransferModel(fieldMHI, fieldV, fieldT, channelSize, dVolume, shape)
