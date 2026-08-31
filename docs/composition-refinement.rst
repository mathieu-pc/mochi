Composition - Refinement Algorithms
===================================

MOCHI features two refinement algorithms,
:py:func:`Mochi.AdaptiveScanline.refineGridToParticleScale` and :py:func:`Mochi.AdaptiveScanline.refineGridToParticleScale`.
These algorithms are designed for SPH and Moving Mesh codes respectively.
Other refinement algorithms can be composed using :py:func:`Mochi.AdaptiveScanline.composeRefinementStrategy`.

:py:func:`Mochi.AdaptiveScanline.composeRefinementStrategy` takes for argument a function which determines the particles a cell should consider and a bisect condition which determines if a cell should be bisected depending on the the mask of its included particles.
For example, :py:func:`Mochi.AdaptiveScanline.refineGridToParticleScale` is composed from :py:func:`Mochi.AdaptiveScanline.intersectIncell` and :py:func:`np.any`.
Therefore, :py:func:`Mochi.AdaptiveScanline.refineGridToParticleScale` bisects cells until their intersecting particles are bigger than them.
