"""
Base Mochi only includes optically thin models
Feel free to write your own
"""
import warnings
import numpy as np

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

def opticallyThin(fieldMHI, fieldV, fieldT, channelSize, dVolume, shape,
		**kwargs
	):
	"""
	Assemble fields into an HI cube using optically thin approximation
	
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
	cube = np.zeros( (nChannel, shape[1], shape[2]) ) * numerator.unit

	spectrumRange = channelSize * (np.arange(nChannel) - (nChannel - 1) / 2)
	diff = fieldV[None, ...] - spectrumRange[:, None, None, None]
	gaussians = np.exp(-diff**2 / (2 * fieldT[None, ...]))
	cube = np.sum(numerator[None, ...] * gaussians, axis=1)  # sum over LOS axis
	cube = np.flip(np.moveaxis(cube, 1, 2), axis=2)
	return cube


def adaptiveOpticallyThin(fieldMHI, fieldV, fieldT, channelSize, cellVolume, cubeShape,
		*,
		cubeFieldIndices = None,
		defaultRadiativeModel = opticallyThin,
		**kwargs
	):
	"""
	Assemble fields into an HI cube using optically thin approximation.
	This version leverages adaptive cube for optimisation.
	
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
	cubeShape:
		spatial shape of cube

	Returns
	-------
	mock cube
	"""
	if cubeFieldIndices is None:		
		warnings.warn("cubeFieldIndices is expected, will attempt defaulting to " + defaultRadiativeModel.__name__, UserWarning)
		return defaultRadiativeModel(fieldMHI, fieldV, fieldT, channelSize, cellVolume, cubeShape, **kwargs)
	nChannel = getChannelNumber(fieldV, fieldMHI, fieldT, channelSize)
	spectrumRange = (channelSize * (np.arange(nChannel) - (nChannel-1)/2))
	fieldT[fieldMHI==0] = 1 * fieldT.unit
	numerator = fieldMHI / np.sqrt(2*np.pi*fieldT) * channelSize * cellVolume
	diff = fieldV[..., None] - spectrumRange[None, :]
	fieldSpectrum = numerator[...,None] * np.exp(-diff**2 / (2 * fieldT[...,None]))
	diff = fieldV[None, ...] - spectrumRange[:, None]
	fieldSpectrum = numerator * np.exp(-diff**2 / (2 * fieldT[None, ...]))
	hyperCube = fieldSpectrum[:, cubeFieldIndices.flatten()]
	cube = np.sum(hyperCube.reshape(nChannel, *cubeShape), axis = 1)
	return np.flip(np.moveaxis(cube, 1, 2), axis = 2)
