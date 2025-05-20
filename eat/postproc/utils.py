from typing import Iterator
import os
from typing import Dict, Tuple

# alias for “name -> (x, y)”
Coordinates = Dict[str, Tuple[float, float]]

def find_uvfits(
    base_path: str,
    extension: str = ".uvfits",
    avg_suffix: str = "+avg",
    include_avg: bool = False,
    recursive: bool = False
) -> Iterator[str]:
    """
    Yield files under `base_path` that end with `extension`, optionally recursing into subdirectories.
    By default, skips those ending in `avg_suffix` before the extension.

    Args:
        base_path: Directory in which to look.
        extension: File extension to match (default: '.uvfits').
                   May be provided with or without leading dot.
        avg_suffix: The string that precedes the extension for "averaged" files (default: '+avg').
        include_avg: If True, include files ending in '{avg_suffix}{extension}'; if False, skip them.
        recursive: If True, walk into subdirectories; otherwise only the top-level directory.

    Yields:
        Full path (as a string) to each matching file.
    """
    # Ensure extension begins with a dot
    ext = extension if extension.startswith('.') else f".{extension}"
    avg_full = f"{avg_suffix}{ext}"

    # Inner function to handle recursion
    def scan_dir(path: str) -> Iterator[str]:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    if recursive:
                        yield from scan_dir(entry.path)
                    continue
                name = entry.name
                if not name.endswith(ext):
                    continue
                if not include_avg and name.endswith(avg_full):
                    continue
                yield entry.path

    # Start scanning from the base path and all paths 
    yield from scan_dir(base_path)
