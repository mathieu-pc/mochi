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
	numerator = fieldM / np.sqrt(2 * np.pi * fieldT) * channelWidth * cellsVolume
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


def _calculateFieldSpectrum(fieldM, fieldV, fieldT, channelWidth, nSigma = 5):
	nChannel = getChannelNumber(fieldV, fieldM, fieldT, channelWidth)
	spectrumRange = (np.arange(nChannel) - (nChannel-1)/2).reshape(nChannel).astype(int)


	channelDispersion = np.array((fieldT / channelWidth ** 2).decompose())
	span = np.sqrt(channelDispersion) * nSigma
	mean = np.array((fieldV / channelWidth).decompose())

	channelRanges = np.column_stack(((mean - span).astype(int), (mean + span + 0.6).astype(int)))
	np.clip(channelRanges, spectrumRange[0], spectrumRange[-1], out = channelRanges)
	centralChannel = (mean).astype(int)
	np.clip(centralChannel, spectrumRange[0], spectrumRange[-1], out = centralChannel)

	difference = channelRanges[:,1] - channelRanges[:,0]
	cumulative = np.cumulative_sum(difference, include_initial = True)

	cellIndices = np.arange(len(fieldM))
	cellIndices = np.repeat(cellIndices, difference)

	cellRanges = np.column_stack((cumulative[:-1], cumulative[1:]))
	cellStartIndices = cumulative[:-1]
	cellEndIndices = cumulative[1:]

	channelDiff = np.arange(len(cellIndices)) - cumulative[cellIndices] - (difference[cellIndices] - difference[cellIndices] % 2) / 2
	fieldSpectrum = np.exp( - channelDiff ** 2 / (2 * channelDispersion[cellIndices]))
	cumulativeSpectrum = np.cumulative_sum(fieldSpectrum, include_initial = True)
	weights = channelDispersion
	weights[weights != 0] = 1 / np.sqrt(2 * np.pi * channelDispersion[weights != 0])
	weights *= fieldM.value
	fieldSpectrum *= weights[cellIndices]

	offset = np.max(np.abs(channelRanges))
	channelRanges += offset

	return fieldSpectrum * fieldM.unit, cellRanges, channelRanges


def adaptiveOpticallyThin(fieldMHI, fieldV, fieldT, channelWidth, cellVolume, volumeShape, cells = None, cellUnit = dimensionless_unscaled, *, indexType = np.uintc, defaultRenderer = opticallyThin, **kwargs):
	if cells is None:
		warnings.warn("cells is expected, will attempt defaulting to " + defaultRenderer.__name__, UserWarning)
		cube = defaultRenderer(fieldMHI, fieldV, fieldT, channelWidth, cellsVolume, volumeShape, **kwargs)
		return cube
	xyz0 = np.min(cells, axis = 0)
	dx = xyz0[-1]
	xyz0[-1] = 0
	N = len(cells)
	cellVolumes = cellVolume * cells[:, -1] / dx  # scaling by depth axis which is integrated
	cellRange = np.arange(N, dtype = indexType)
	cellsBegin = np.round((cells[:,:-1] - xyz0[:-1])/dx).astype(indexType)
	cellsFinish = np.round((cells[:,:-1] - xyz0[:-1] + cells[:,-1][:,np.newaxis])/dx).astype(indexType)
	fieldSpectra, cellRanges, channelRanges = _calculateFieldSpectrum(fieldMHI * cellVolumes, fieldV, fieldT, channelWidth)
	cubeUnit = fieldSpectra.unit
	fieldSpectra = fieldSpectra[:, None, None].value
	cube = np.zeros((np.max(channelRanges[:,1]), volumeShape[1], volumeShape[2]))
	for i in cellRange:
		x_start, y_start, z_start = cellsBegin[i]
		x_end, y_end, z_end = cellsFinish[i]
		cube[channelRanges[i,0]:channelRanges[i,1], y_start:y_end, z_start:z_end] += fieldSpectra[cellRanges[i,0]:cellRanges[i,1]]
	cube = np.flip(np.moveaxis(cube, 1, 2), axis = 2) * cubeUnit
	return cube
