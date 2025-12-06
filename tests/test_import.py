import importlib


def test_import_package():
    """Basic import checks for the package top-level module.

    The repository exposes the package as `GalCubeCraft` (see `src/GalCubeCraft`).
    Ensure the package has the convenience `init` function and exposes the
    main class name in the module namespace.
    """
    mod = importlib.import_module('GalCubeCraft')
    # convenience constructor (defined in src/GalCubeCraft/__init__.py)
    assert hasattr(mod, 'init')
    # top-level class is provided from core.py
    assert hasattr(mod, 'GalCubeCraft')


def test_core_utils_visualise_modules():
    """Import internal modules and check for a handful of expected symbols."""
    core = importlib.import_module('GalCubeCraft.core')
    assert hasattr(core, 'GalCubeCraft')

    utils = importlib.import_module('GalCubeCraft.utils')
    # utils should provide beam convolution helper
    assert hasattr(utils, 'convolve_beam')
    assert hasattr(utils, 'add_beam')

    vis = importlib.import_module('GalCubeCraft.visualise')
    assert hasattr(vis, 'visualise')
