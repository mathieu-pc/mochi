from astropy.io import fits
from astropy import units


def makeCards(cube, pixelSize, channelSize, beam, **kwargs):
	cards = []
	try:
		cards += [fits.Card("BUNIT", str(cube.unit).replace(" ", ""))]
	except:
		pass

	cards += getAxisCards(1, - pixelSize.to(units.deg), (cube.shape[1]+1)/2, 180, "RA---TAN")
	cards += getAxisCards(2, + pixelSize.to(units.deg), (cube.shape[2]+1)/2, 90, "DEC--TAN")
	cards += getAxisCards(3, + channelSize.to(units.m / units.s), (cube.shape[0]+1)/2, 0, "VOPT")

	if not beam is None:
		cards += [fits.Card("BMAJ", beam.major.to(units.deg).value)]
		cards += [fits.Card("BMIN", beam.minor.to(units.deg).value)]
		cards += [fits.Card("BPA", beam.pa.to(units.deg).value)]


	for key in kwargs.keys():
		cards += [fits.Card(key, kwargs[key])]
	return cards

def getAxisCards(axisNumber, delta, referenceIndex, referenceValue, type):
	result = []
	axisStr = str(axisNumber)
	try:
		deltaValue = delta.value
		deltaUnit = str(delta.unit)
	except:
		deltaValue = delta
		deltaUnit = "PAR"
	try:
		refValue = referenceValue.value
	except:
		refValue = referenceValue
	result += [fits.Card("CDELT" + axisStr, deltaValue)]
	result += [fits.Card("CRPIX" + axisStr, referenceIndex)]
	result += [fits.Card("CRVAL" + axisStr, refValue)]
	result += [fits.Card("CTYPE" + axisStr, type)]
	result += [fits.Card("CUNIT" + axisStr, deltaUnit)]
	return result


if __name__ == "__main__":
	import numpy as np
	from radio_beam import Beam
	beam = Beam(30 * units.arcsec, 25 * units.arcsec)
	cube = np.ones( (100,) * 3) * units.Jy / units.beam
	cards = makeCards(cube, 5 * units.arcsec, 4 * units.km / units.s, beam)
	header = fits.Header(cards)
	fits.writeto("cubecards.fits", cube.value, header, overwrite = True)