import importlib

def test_import_package():
    mod = importlib.import_module('cubecraft')
    assert hasattr(mod, '__version__')
    assert hasattr(mod, 'generator')


def test_generator_module():
    gm = importlib.import_module('cubecraft.generator')
    assert hasattr(gm, 'ResolvedSpectralCubeDataset')
    assert hasattr(gm, 'FinalSpectralCubeDataset')
