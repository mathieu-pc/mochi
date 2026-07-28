User Guide
==========

This is my handwritten documentation.

Installation
------------

Install the package with:

.. code-block:: bash

   pip install "git+https://github.com/mathieu-pc/mochi.git"


Running MOCHI
-------------

MOCHI converts simulation particle data into mock datacubes.

This is handled by :py:func:`Mochi.Interface.makeCube`.
Additionally, the wrapper :py:func:`Mochi.Interface.makeCubeFromSource` allows the creation of mock cubes from :class:`martini.sources.sph_source.SPHSource` objects.
MARTINI offers a great simulation to mock interface. Please refer to its documentation for help.

Demo files are included in the github parent directory.