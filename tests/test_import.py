import os
import ast


def _read_module_source(path):
    with open(path, 'r', encoding='utf8') as f:
        return f.read()


def test_package_files_exist():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(root, 'src', 'GalCubeCraft')
    init_py = os.path.join(src_dir, '__init__.py')
    assert os.path.isdir(src_dir), f"Package folder not found: {src_dir}"
    assert os.path.isfile(init_py), f"__init__.py not found in package: {init_py}"


def test_init_exports():
    """Lightweight checks on `__init__.py` without importing heavy deps.

    We parse the source of `__init__.py` to ensure it exposes the convenience
    `init` function and references the main `GalCubeCraft` symbol (import or re-export).
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    init_path = os.path.join(root, 'src', 'GalCubeCraft', '__init__.py')
    src = _read_module_source(init_path)

    # Quick textual sanity checks (keeps tests fast and avoids importing heavy libraries)
    assert 'def init' in src or 'def init(' in src
    assert 'GalCubeCraft' in src


def test_utils_and_visualise_have_expected_defs():
    """Parse `utils.py` and `visualise.py` AST to check for expected function defs.
    This avoids importing the modules which may require heavy optional deps.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    utils_path = os.path.join(root, 'src', 'GalCubeCraft', 'utils.py')
    vis_path = os.path.join(root, 'src', 'GalCubeCraft', 'visualise.py')

    utils_src = _read_module_source(utils_path)
    vis_src = _read_module_source(vis_path)

    utils_tree = ast.parse(utils_src)
    vis_tree = ast.parse(vis_src)

    utils_funcs = {n.name for n in ast.walk(utils_tree) if isinstance(n, ast.FunctionDef)}
    vis_funcs = {n.name for n in ast.walk(vis_tree) if isinstance(n, ast.FunctionDef)}

    assert 'convolve_beam' in utils_funcs
    assert 'add_beam' in utils_funcs
    assert 'visualise' in vis_funcs or 'visualise' in vis_src
