#!/usr/bin/env python3
"""
link_hops_scans.py

Implementation of the Mk4 file staging functionality which used to be
handled by ehthops's original 0.bootstrap/bin/2.link for better performance
since this program spawns zero `find`/`grep`/`sed` subprocesses.

Configuration is read from environment variables, mirroring the shell
variables required by the original 2.link script:
SRCDIR, DATADIR, CORRDAT, FILTERSTRING (optional), HAXP ("true"/else)
"""

import fnmatch
import os
import re
import shutil
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

ROOTFILE_REGEX = re.compile(r'^[a-zA-Z0-9+_-]+\.[a-zA-Z0-9]{6}$')
ROOTFILE_EXT_REGEX = re.compile(r'[a-zA-Z0-9]{6}$')
EXPT_NO_REGEX = re.compile(r'^[0-9]{4,5}$')
FRINGEFILE_REGEX = re.compile(r'[A-Za-z]{2}\.[A-Za-z]\.[0-9]+\.[A-Za-z0-9]{6}$')
CORRDAT_SPLIT_REGEX = re.compile(r'[ \t\n:,]+')


def split_corrdat(s):
    """
    Split a CORRDAT string into its constituent directory names, using
    the same whitespace/colon/comma splitting rules as the original 2.link.

    Args:
        s (str): The CORRDAT string to split.

    Returns:
        list: A list of directory names extracted from the CORRDAT string.
    """
    return [d for d in CORRDAT_SPLIT_REGEX.split(s) if d]


def list_files(path, follow_symlinks=True):
    """
    List the basenames of regular files (after symlink resolution) directly
    inside `path`, in arbitrary order.

    Args:
        path (str): The directory path to list files from.
        follow_symlinks (bool): Whether to follow symlinks when checking if an entry
            is a regular file. Defaults to True.

    Returns:
        list: A list of basenames of regular files directly inside `path`.
    """
    try:
        with os.scandir(path) as it:
            entries = list(it)
    except OSError:
        return []
    out = []
    for e in entries:
        try:
            if e.is_file(follow_symlinks=follow_symlinks):
                out.append(e.name)
        except OSError:
            continue  # e.g. broken symlink; find -L would skip it too
    return out


def list_entries_raw(path):
    """
    Basenames of directory entries directly inside `path`, without any filtering.

    Args:
        path (str): The directory path to list entries from.

    Returns:
        list: A list of basenames of directory entries directly inside `path`.
    """
    try:
        return os.listdir(path)
    except OSError:
        return []


def walk_files(root):
    """
    All regular files (after symlink resolution) recursively under `root`,
    in arbitrary order.

    Args:
        root (str): The root directory to start the search from.

    Yields:
        str: The full path of each regular file found under `root`.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if os.path.isfile(full):
                yield full


def replace_last(scandir, old, new):
    """
    Replace the last occurrence of `old` in `scandir` with `new`.

    Args:
        scandir (str): The original string.
        old (str): The substring to be replaced.
        new (str): The substring to replace with.

    Returns:
        str: The modified string with the last occurrence of `old` replaced by `new`.
    """
    idx = scandir.rfind(old)
    if idx == -1:
        return scandir
    return scandir[:idx] + new + scandir[idx + len(old):]


def link(src, dest):
    """
    Create a symbolic link from `src` to `dest`, logging a warning if
    the link already exists (mirroring the original 2.link's `ln -s`
    behavior). Does not overwrite existing files or directories.

    Args:
        src (str): The source file path to link from.
        dest (str): The destination file path to link to.
    """
    try:
        os.symlink(src, dest)
    except FileExistsError:
        logging.warning(f"ln: failed to create symbolic link '{dest}': File exists")


def main():
    srcdir = os.environ.get('SRCDIR', '')
    datadir = os.environ.get('DATADIR', '')
    corrdat = os.environ.get('CORRDAT', '')
    filterstring = os.environ.get('FILTERSTRING', '')
    haxp = os.environ.get('HAXP', '') == 'true'

    for corrdir in split_corrdat(corrdat):
        prev_src_scan_dir = None
        base_dir = os.path.join(srcdir, corrdir)

        if not os.path.isdir(base_dir):
            logging.warning(f"find: '{base_dir}': No such file or directory")
            continue

        for root_file in walk_files(base_dir):
            basename = os.path.basename(root_file)

            if not ROOTFILE_REGEX.match(basename):
                continue

            # Skip files if the path contains the pattern "haxp/"
            if 'haxp/' in root_file:
                continue

            # Skip files if the path does not match $FILTERSTRING
            # (default is "[-_/]$BAND[-_/]". This is a *glob* pattern,
            # not a regex, so we use fnmatch.fnmatch() instead of re.search()
            if filterstring and not fnmatch.fnmatch(root_file, '*' + filterstring + '*'):
                continue

            src_scan_dir = os.path.dirname(root_file)
            src_expt_dir = os.path.dirname(src_scan_dir)

            if not EXPT_NO_REGEX.match(os.path.basename(src_expt_dir)):
                logging.info(f"Skipping {src_expt_dir} with {root_file}")
                continue

            logging.info(root_file)

            dest_expt_dir = os.path.join(datadir, os.path.basename(src_expt_dir))
            dest_scan_dir = os.path.join(dest_expt_dir, os.path.basename(src_scan_dir))

            extension = root_file[-6:]

            if os.path.isdir(dest_scan_dir):
                if src_scan_dir == prev_src_scan_dir:
                    dest_ext_set = set()
                    for name in list_files(dest_scan_dir):
                        m = ROOTFILE_EXT_REGEX.search(name)
                        if m:
                            dest_ext_set.add(m.group())
                    dest_extensions = sorted(dest_ext_set)

                    # If the destination scan directory already exists, check if the current extension
                    # is <= the maximum existing extension. If so, skip linking this scan. Otherwise,
                    # remove the existing destination scan directory and proceed to link the new scan.
                    if dest_extensions:
                        max_dest_extension = dest_extensions[-1]
                        if extension <= max_dest_extension:
                            logging.info(f"Skipping {extension} in favour of {max_dest_extension}")
                            prev_src_scan_dir = src_scan_dir
                            continue
                        else:
                            logging.info("Replacing " + "\n".join(dest_extensions) +
                                  f" with extension {extension}")
                            shutil.rmtree(dest_scan_dir)
                else:
                    prev_src_scan_dir = src_scan_dir
                    continue

            os.makedirs(dest_scan_dir, exist_ok=True)

            # Link all files in the source scan directory to the destination scan directory,
            # except for existing fringefiles (if any) in the source scan directory.
            for name in list_files(src_scan_dir):
                if name.endswith('.' + extension) and not FRINGEFILE_REGEX.search(name):
                    link(os.path.join(src_scan_dir, name),
                         os.path.join(dest_scan_dir, name))

            prev_src_scan_dir = src_scan_dir

            # If HAXP=true then replace ALMA data with the contents of "haxp/" directories
            if haxp:
                haxp_src_scan_dir = replace_last(src_scan_dir, 'hops/', 'haxp/')

                for name in list_entries_raw(dest_scan_dir):
                    if name.startswith('A'):
                        path = os.path.join(dest_scan_dir, name)
                        try:
                            os.remove(path)
                        except IsADirectoryError:
                            logging.warning(f"rm: cannot remove '{path}': Is a directory")
                        except FileNotFoundError:
                            pass

                if not os.path.isdir(haxp_src_scan_dir):
                    logging.warning(f"find: '{haxp_src_scan_dir}': No such file or directory")
                else:
                    for name in list_files(haxp_src_scan_dir):
                        if not FRINGEFILE_REGEX.search(name):
                            link(os.path.join(haxp_src_scan_dir, name),
                                 os.path.join(dest_scan_dir, name))

    return 0


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
