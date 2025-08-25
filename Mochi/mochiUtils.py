def _astropyUnitWrap(cube):
	try:
		return cube.value.astype(float), cube.unit
	except:
		return cube.astype(float), 1
