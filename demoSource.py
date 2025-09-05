"""
Create a set of particles in Mochi/martini dictionary format.
The source created is unphysical and for illustrative purposes only.
Borrows heavily from martini's demo_source.
"""

import numpy as np
from astropy import units
from scipy.spatial import KDTree

def calculateNearestNeighbourDistance(xyz):
	dist, _ = KDTree(xyz).query(xyz, k = 2)
	return dist[:,1]

def demoSource(N=500):
	"""
	Create a set of particles ressembling a galaxy.
	This is not meant to be representative of any actual galaxy model.
	This borrows from martini's demo_source but I wanted to add spirals.

	Parameters
	----------
	N : int
		Number of particles to generate in source.

	Returns
	-------
	out : dictionary
		A dictionary of particles.
	"""
	phi = np.random.rand(N) * 2 * np.pi
	r = np.abs(np.random.normal(3, 1.5, N))

	x = np.random.rand(N) - 0.5
	r /= np.sqrt(1-0.5 * np.cos(phi - 1.5 * (r-1))**2) #density wave
	y = r * np.cos(phi) + np.random.normal(0, 0.2, N)
	z = r * np.sin(phi) + np.random.normal(0, 0.2, N)
	xyz_g = np.vstack((x, y, z)) * units.kpc
	# arctan rotation curve
	vphi = 50 * np.arctan(r)
	vx = -vphi * np.sin(phi)
	vy = vphi * np.cos(phi)
	# small random z velocities
	vz = (np.random.rand(N) * 2.0 - 1.0) * 5
	vxyz_g = np.vstack((vx, vy, vz)) * units.km * units.s**-1
	T_g = 20 * np.ones(N) * (units.km / units.s)**2
	# HI masses with some scatter
	mHI_g = np.ones(N) + 0.01 * (np.random.rand(N) - 0.5)
	mHI_g = mHI_g / mHI_g.sum() * 5.0e9 * units.Msun
	# Smoothing lengths based on nearest neighbour distance
	xyz_g = np.moveaxis(xyz_g, 0, -1)
	hsm_g = 2 * calculateNearestNeighbourDistance(xyz_g.value) * xyz_g.unit
	mask = hsm_g < 0.5 * units.kpc
	hsm_g[mask] = 0.5 * units.kpc
	vxyz_g = np.moveaxis(vxyz_g, 0, -1)
	particles = {
		"m": mHI_g,
		"T_g": T_g,
		"mHI_g": mHI_g,
		"xyz_g": xyz_g,
		"vxyz_g": vxyz_g,
		"hsm_g": hsm_g
	}
	return particles

if __name__ == "__main__":
	particles = demoSource()
	from matplotlib import pyplot as plt
	plt.plot(
		particles["xyz_g"][:,1].value,
		particles["xyz_g"][:,2].value,
		'ko',
		markersize = 5,
		alpha = 0.1,
		markeredgewidth=0.0
	)
	plt.show()
