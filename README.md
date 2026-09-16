# MOCHI

![Tests](https://github.com/mathieu-pc/mochi/actions/workflows/test.yaml/badge.svg)

Mock Observation Cubes of HI (MOCHI)

[Documentation](https://astromochi.readthedocs.io/en/latest/)

MOCHI is a mock imaging software that converts hydro-dynamical simulation data into high fidelity radio-astronomical data-cubes.

MOCHI applies an intermediate step where the relevant fields (density, temperature, velocity) are interpolated in simulation space before being collapsed into a data-cube. Adaptive resolution allocation and other optimizations allow MOCHI to achieve high accuracy and speed.

Install:
```
pip install "git+https://github.com/mathieu-pc/mochi.git"
```

For support, email 21mpc3@queensu.ca

Cite: https://iopscience.iop.org/article/10.3847/1538-3881/ada567
