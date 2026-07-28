"""
Base Mochi only includes optically thin models
Feel free to write your own
"""
import warnings
import numpy as np
from astropy.units import dimensionless_unscaled


def getChannelNumber(VX, M, T, channelWidth, *, minChannelNumber = 120, maxChannelNumber = 300):
	"""
	Utility function
	Estimates the best number of channels to get most of the cube flux in.
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
	guess = int((np.abs(v[i]) + 3 * np.sqrt(t[i]))/channelWidth + 1 )
	channelNumber = max(min((guess + 25)*2, maxChannelNumber), minChannelNumber)+1
	return channelNumber


def calculateFieldSpectrum(fieldM, fieldV, fieldT, cellsVolume, channelWidth):
	nChannel = getChannelNumber(fieldV, fieldM, fieldT, channelWidth)
	spectrumRange = (channelWidth * (np.arange(nChannel) - (nChannel-1)/2))
	fieldT[fieldM==0] = 1 * fieldT.unit
	numerator = fieldM / np.sqrt(2*np.pi*fieldT) * channelWidth * cellsVolume
	diff = fieldV[None, ...] - spectrumRange[:, None]
	fieldSpectrum = numerator * np.exp(-diff**2 / (2 * fieldT[None, ...]))
	return fieldSpectrum


def opticallyThin(fieldMHI, fieldV, fieldT, channelWidth, dVolume, volumeShape,
		**kwargs
	):
	"""
	Assemble fields into an HI cube using optically thin approximation
	"""
	nChannel = getChannelNumber(fieldV, fieldMHI, fieldT, channelWidth)
	spectrumRange = (channelWidth * (np.arange(nChannel) - (nChannel-1)/2)).reshape(nChannel, 1, 1, 1)
	fieldMHI = fieldMHI.reshape(volumeShape)
	fieldT = fieldT.reshape(volumeShape)
	fieldV = fieldV.reshape(volumeShape)
	fieldT[fieldMHI==0] = 1 * fieldT.unit
	numerator = fieldMHI / np.sqrt(2*np.pi*fieldT) * channelWidth * dVolume
	cube = np.zeros( (nChannel, volumeShape[1], volumeShape[2]) ) * numerator.unit
	spectrumRange = channelWidth * (np.arange(nChannel) - (nChannel - 1) / 2)
	diff = fieldV[None, ...] - spectrumRange[:, None, None, None]
	gaussians = np.exp(-diff**2 / (2 * fieldT[None, ...]))
	cube = np.sum(numerator[None, ...] * gaussians, axis=1)  # sum over LOS axis
	cube = np.flip(np.moveaxis(cube, 1, 2), axis=2)
	return cube


def adaptiveOpticallyThin(fieldMHI, fieldV, fieldT, channelWidth, cellsVolume, volumeShape, cells = None, cellUnit = dimensionless_unscaled, *, indexType = np.uintc, defaultRenderer = opticallyThin, **kwargs):
	if cells is None:
		warnings.warn("cells is expected, will attempt defaulting to " + defaultRenderer.__name__, UserWarning)
		cube = defaultRenderer(fieldMHI, fieldV, fieldT, channelWidth, cellsVolume, volumeShape, **kwargs)
		return cube
	xyz0 = np.min(cells, axis = 0)
	dx = xyz0[-1]
	elementVolume = dx ** 3 * cellUnit ** 3
	xyz0[-1] = 0
	N = len(cells)
	cellRange = np.arange(N, dtype = indexType)
	cellsBegin = np.round((cells[:,:-1] - xyz0[:-1])/dx).astype(indexType)
	cellsFinish = np.round((cells[:,:-1] - xyz0[:-1] + cells[:,-1][:,np.newaxis])/dx).astype(indexType)
	fieldSpectra = calculateFieldSpectrum(fieldMHI, fieldV, fieldT, elementVolume, channelWidth)
	cubeUnit = fieldSpectra.unit
	fieldSpectra *= cellsFinish[:,0] - cellsBegin[:,0]
	fieldSpectra = fieldSpectra[:,:,None,None].value
	cube = np.zeros((fieldSpectra.shape[0], volumeShape[1], volumeShape[2]))
	for i in cellRange:
		x_start, y_start, z_start = cellsBegin[i]
		x_end, y_end, z_end = cellsFinish[i]
		cube[:,y_start:y_end, z_start:z_end] += fieldSpectra[:,i]
	cube = np.flip(np.moveaxis(cube, 1, 2), axis = 2) * cubeUnit
	return cube