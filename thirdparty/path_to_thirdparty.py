import os.path as path
import sys

HERE_PATH = path.normpath(path.dirname(__file__))

SUBMODULES = {
    "pi3": path.join("pi3", "pi3", "models"),
    "doppelgangers-plusplus": path.join("doppelgangers-plusplus", "mast3r"),
    path.join("doppelgangers-plusplus", "dust3r"): path.join(
        "doppelgangers-plusplus", "dust3r", "dust3r"
    ),
    path.join("doppelgangers-plusplus", "dust3r", "croco"): path.join(
        "doppelgangers-plusplus", "dust3r", "croco", "models"
    ),
    "vggt": path.join("vggt", "vggt", "models"),
    "mapanything": path.join("mapanything", "mapanything", "models"),
}

# Submodules using a src/ layout (package lives at thirdparty/<name>/src/<pkg>/).
# Value is (check_relative_path, src_relative_path).
SRC_LAYOUT_SUBMODULES = {
    "dvlt": (
        path.join("dvlt", "src", "dvlt", "model"),  # existence check
        path.join("dvlt", "src"),                     # added to sys.path
    ),
}

# vggsfm lives directly under thirdparty/ (not a git submodule) and uses
# relative imports, so we add HERE_PATH itself so it is importable as a package.
PACKAGES = {
    "vggsfm": path.join("vggsfm", "track_modules"),
}

for name, check_path in SUBMODULES.items():
    repo_path = path.normpath(path.join(HERE_PATH, name))
    full_check = path.join(HERE_PATH, check_path)
    if path.exists(full_check):
        sys.path.insert(0, repo_path)
    else:
        raise ImportError(
            f"{name} is not initialized, could not find: {full_check}.\n "
            "Did you forget to run 'git submodule update --init --recursive' ?"
        )

for name, (check_rel, src_rel) in SRC_LAYOUT_SUBMODULES.items():
    full_check = path.join(HERE_PATH, check_rel)
    src_path = path.normpath(path.join(HERE_PATH, src_rel))
    if path.exists(full_check):
        sys.path.insert(0, src_path)
    else:
        raise ImportError(
            f"{name} is not initialized, could not find: {full_check}.\n "
            "Did you forget to run 'git submodule update --init --recursive' ?"
        )

for name, check_path in PACKAGES.items():
    full_check = path.join(HERE_PATH, check_path)
    if path.exists(full_check):
        if HERE_PATH not in sys.path:
            sys.path.insert(0, HERE_PATH)
    else:
        raise ImportError(
            f"{name} is not initialized, could not find: {full_check}."
        )
