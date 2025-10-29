"""
Base Mochi only includes optically thin models
Feel free to write your own
"""
import warnings
import numpy as np
from astropy.units import dimensionless_unscaled


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


def calculateFieldSpectrum(fieldM, fieldV, fieldT, cellVolume, channelSize):
	nChannel = getChannelNumber(fieldV, fieldM, fieldT, channelSize)
	spectrumRange = (channelSize * (np.arange(nChannel) - (nChannel-1)/2))
	fieldT[fieldM==0] = 1 * fieldT.unit
	numerator = fieldM / np.sqrt(2*np.pi*fieldT) * channelSize * cellVolume
	diff = fieldV[None, ...] - spectrumRange[:, None]
	fieldSpectrum = numerator * np.exp(-diff**2 / (2 * fieldT[None, ...]))
	return fieldSpectrum


def opticallyThin(fieldMHI, fieldV, fieldT, channelSize, dVolume, volumeShape,
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
	volumeShape:
		spatial shape of fieldMHI, fieldV, fieldT

	Returns
	-------
	mock cube
	"""
	nChannel = getChannelNumber(fieldV, fieldMHI, fieldT, channelSize)
	spectrumRange = (channelSize * (np.arange(nChannel) - (nChannel-1)/2)).reshape(nChannel, 1, 1, 1)
	fieldMHI = fieldMHI.reshape(volumeShape)
	fieldT = fieldT.reshape(volumeShape)
	fieldV = fieldV.reshape(volumeShape)
	fieldT[fieldMHI==0] = 1 * fieldT.unit
	numerator = fieldMHI / np.sqrt(2*np.pi*fieldT) * channelSize * dVolume
	cube = np.zeros( (nChannel, volumeShape[1], volumeShape[2]) ) * numerator.unit
	spectrumRange = channelSize * (np.arange(nChannel) - (nChannel - 1) / 2)
	diff = fieldV[None, ...] - spectrumRange[:, None, None, None]
	gaussians = np.exp(-diff**2 / (2 * fieldT[None, ...]))
	cube = np.sum(numerator[None, ...] * gaussians, axis=1)  # sum over LOS axis
	cube = np.flip(np.moveaxis(cube, 1, 2), axis=2)
	return cube


def adaptiveOpticallyThin(fieldMHI, fieldV, fieldT, channelSize, cellVolume, volumeShape, cells = None, cellUnit = dimensionless_unscaled, *, indexType = np.uintc, defaultRenderer = opticallyThin, **kwargs):
	if cells is None:
		warnings.warn("cells is expected, will attempt defaulting to " + defaultRenderer.__name__, UserWarning)
		return defaultRenderer(fieldMHI, fieldV, fieldT, channelSize, cellVolume, volumeShape, **kwargs)
	xyz0 = np.min(cells, axis = 0)
	dx = xyz0[-1]
	elementVolume = dx ** 3 * cellUnit ** 3
	xyz0[-1] = 0
	N = len(cells)
	cellRange = np.arange(N, dtype = indexType)
	cellsBegin = np.round((cells[:,:-1] - xyz0[:-1])/dx).astype(indexType)
	cellsFinish = np.round((cells[:,:-1] - xyz0[:-1] + cells[:,-1][:,np.newaxis])/dx).astype(indexType)
	fieldSpectra = calculateFieldSpectrum(fieldMHI, fieldV, fieldT, elementVolume, channelSize)
	cubeUnit = fieldSpectra.unit
	fieldSpectra *= cellsFinish[:,0] - cellsBegin[:,0]
	fieldSpectra = fieldSpectra[:,:,None,None].value
	cube = np.zeros((fieldSpectra.shape[0], volumeShape[1], volumeShape[2]))
	for i in cellRange:
		x_start, y_start, z_start = cellsBegin[i]
		x_end, y_end, z_end = cellsFinish[i]
		cube[:,y_start:y_end, z_start:z_end] += fieldSpectra[:,i]
	return np.flip(np.moveaxis(cube, 1, 2), axis = 2) * cubeUnit