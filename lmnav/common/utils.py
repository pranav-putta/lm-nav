"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import gc
import torch.distributed
import tracemalloc
import json
import logging
import os
import pickle
import re
import shutil
import urllib
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from lmnav.common.registry import registry
from torch.utils.model_zoo import tqdm
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    extract_archive,
)
import linecache
from torch import nn
import torch.nn.functional as F

import time
from contextlib import contextmanager


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def get_cache_path(rel_path):
    return os.path.expanduser(os.path.join(registry.get_path("cache_root"), rel_path))


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


# The following are adapted from torchvision and vissl
# torchvision: https://github.com/pytorch/vision
# vissl: https://github.com/facebookresearch/vissl/blob/main/vissl/utils/download.py


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        print(f"Error creating directory: {dir_path}")
    return is_success


def get_redirected_url(url: str):
    """
    Given a URL, returns the URL it redirects to or the
    original URL in case of no indirection
    """
    import requests

    with requests.Session() as session:
        with session.get(url, stream=True, allow_redirects=True) as response:
            if response.history:
                return response.url
            else:
                return url


def to_google_drive_download_url(view_url: str) -> str:
    """
    Utility function to transform a view URL of google drive
    to a download URL for google drive
    Example input:
        https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view
    Example output:
        https://drive.google.com/uc?export=download&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp
    """
    splits = view_url.split("/")
    assert splits[-1] == "view"
    file_id = splits[-2]
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_google_drive_url(url: str, output_path: str, output_file_name: str):
    """
    Download a file from google drive
    Downloading an URL from google drive requires confirmation when
    the file of the size is too big (google drive notifies that
    anti-viral checks cannot be performed on such files)
    """
    import requests

    with requests.Session() as session:
        # First get the confirmation token and append it to the URL
        with session.get(url, stream=True, allow_redirects=True) as response:
            for k, v in response.cookies.items():
                if k.startswith("download_warning"):
                    url = url + "&confirm=" + v

        # Then download the content of the file
        with session.get(url, stream=True, verify=True) as response:
            makedir(output_path)
            path = os.path.join(output_path, output_file_name)
            total_size = int(response.headers.get("Content-length", 0))
            with open(path, "wb") as file:
                from tqdm import tqdm

                with tqdm(total=total_size) as progress_bar:
                    for block in response.iter_content(
                        chunk_size=io.DEFAULT_BUFFER_SIZE
                    ):
                        file.write(block)
                        progress_bar.update(len(block))


def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "vissl"})
        ) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
                                  If None, use the basename of the URL.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir(root)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # expand redirect chain if needed
    url = get_redirected_url(url)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead."
                " Downloading " + url + " to " + fpath
            )
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def cache_url(url: str, cache_dir: str) -> str:
    """
    This implementation downloads the remote resource and caches it locally.
    The resource will only be downloaded if not previously requested.
    """
    parsed_url = urlparse(url)
    dirname = os.path.join(cache_dir, os.path.dirname(parsed_url.path.lstrip("/")))
    makedir(dirname)
    filename = url.split("/")[-1]
    cached = os.path.join(dirname, filename)
    with file_lock(cached):
        if not os.path.isfile(cached):
            logging.info(f"Downloading {url} to {cached} ...")
            cached = download(url, dirname, filename=filename)
    logging.info(f"URL {url} cached in {cached}")
    return cached


# TODO (prigoyal): convert this into RAII-style API
def create_file_symlink(file1, file2):
    """
    Simply create the symlinks for a given file1 to file2.
    Useful during model checkpointing to symlinks to the
    latest successful checkpoint.
    """
    try:
        if g_pathmgr.exists(file2):
            g_pathmgr.rm(file2)
        g_pathmgr.symlink(file1, file2)
    except Exception as e:
        logging.info(f"Could NOT create symlink. Error: {e}")


def save_file(data, filename, append_to_json=True, verbose=True):
    """
    Common i/o utility to handle saving data to various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    Specifically for .json, users have the option to either append (default)
    or rewrite by passing in Boolean value to append_to_json.
    """
    if verbose:
        logging.info(f"Saving data to file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "wb") as fopen:
            pickle.dump(data, fopen, pickle.HIGHEST_PROTOCOL)
    elif file_ext == ".npy":
        with g_pathmgr.open(filename, "wb") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        if append_to_json:
            with g_pathmgr.open(filename, "a") as fopen:
                fopen.write(json.dumps(data, sort_keys=True) + "\n")
                fopen.flush()
        else:
            with g_pathmgr.open(filename, "w") as fopen:
                fopen.write(json.dumps(data, sort_keys=True) + "\n")
                fopen.flush()
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "w") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    else:
        raise Exception(f"Saving {file_ext} is not supported yet")

    if verbose:
        logging.info(f"Saved data to file: {filename}")


def load_file(filename, mmap_mode=None, verbose=True, allow_pickle=False):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    if verbose:
        logging.info(f"Loading data from file: {filename}")

    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".txt":
        with g_pathmgr.open(filename, "r") as fopen:
            data = fopen.readlines()
    elif file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding="latin1")
    elif file_ext == ".npy":
        if mmap_mode:
            try:
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np.load(
                        fopen,
                        allow_pickle=allow_pickle,
                        encoding="latin1",
                        mmap_mode=mmap_mode,
                    )
            except ValueError as e:
                logging.info(
                    f"Could not mmap {filename}: {e}. Trying without g_pathmgr"
                )
                data = np.load(
                    filename,
                    allow_pickle=allow_pickle,
                    encoding="latin1",
                    mmap_mode=mmap_mode,
                )
                logging.info("Successfully loaded without g_pathmgr")
            except Exception:
                logging.info("Could not mmap without g_pathmgr. Trying without mmap")
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np.load(fopen, allow_pickle=allow_pickle, encoding="latin1")
        else:
            with g_pathmgr.open(filename, "rb") as fopen:
                data = np.load(fopen, allow_pickle=allow_pickle, encoding="latin1")
    elif file_ext == ".json":
        with g_pathmgr.open(filename, "r") as fopen:
            data = json.load(fopen)
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "r") as fopen:
            data = yaml.load(fopen, Loader=yaml.FullLoader)
    elif file_ext == ".csv":
        with g_pathmgr.open(filename, "r") as fopen:
            data = pd.read_csv(fopen)
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    return data


def abspath(resource_path: str):
    """
    Make a path absolute, but take into account prefixes like
    "http://" or "manifold://"
    """
    regex = re.compile(r"^\w+://")
    if regex.match(resource_path) is None:
        return os.path.abspath(resource_path)
    else:
        return resource_path


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        logging.info(f"Error creating directory: {dir_path}")
    return is_success


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url


def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        shutil.rmtree(dir)
    logging.info(f"Deleted contents of directory: {dir}")


def get_file_size(filename):
    """
    Given a file, get the size of file in MB
    """
    size_in_mb = os.path.getsize(filename) / float(1024**2)
    return size_in_mb


def display_snapshot(snapshot, key_type="lineno", limit=10):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def all_reduce(is_distributed, device, t):
    if not is_distributed:
        return t

    orig_device = t.device
    t = t.to(device)
    torch.distributed.all_reduce(t)

    t = t.to(device=orig_device)
    return t


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.GroupNorm)):
            l.weight.data = l.weight.data.to(torch.bfloat16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.bfloat16)

    model.apply(_convert_weights_to_fp16)


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def sum_dict(d1, d2):
    return {k: d1[k] + d2[k] for k in d1.keys() | d2.keys()}


def all_gather(q, device, world_size):
    """
    Gathers tensor arrays of different lengths across multiple gpus.
    Assumes that q.shape[1:] is identical across gpus, and only pads dim=0

    Parameters
    ----------
        q : tensor array
        ws : world size
        device : current gpu device

    Returns
    -------
        all_q : list of gathered tensor arrays from all the gpus

    """
    local_size = torch.tensor(q.shape[0], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size)

    q = q.to(device)
    size_diff = max(all_sizes).item() - local_size.item()
    print(size_diff)
    if size_diff:
        padding = torch.zeros((size_diff, *q.shape[1:]), device=device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(world_size)]
    torch.distributed.all_gather(all_qs_padded, q)
    all_qs = torch.cat([q[:size] for q, size in zip(all_qs_padded, all_sizes)])
    return all_qs


def find_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                tensors.append((obj, obj.size()))
        except:
            pass

    return tensors


@contextmanager
def catchtime(name=""):
    torch.cuda.synchronize()
    start = time.time()
    yield lambda: time.time() - start
    torch.cuda.synchronize()
    print(f"TimeIt [{name}]: {time.time() - start:.3f} seconds")


def forward_minibatches(fn_forward, list_of_tensors, max_batch_size):
    tensor = torch.cat(list_of_tensors, dim=0)
    return torch.split(
        torch.cat(
            [
                fn_forward(tensor[i : i + max_batch_size])
                for i in range(0, tensor.shape[0], max_batch_size)
            ],
            dim=0,
        ),
        [len(t) for t in list_of_tensors],
    )


def levenshtein_distance(tensor1, tensor2):
    """
    Compute the Levenshtein distance between two tensors.
    """
    m, n = tensor1.size(0), tensor2.size(0)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int)

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i, j] = j
            elif j == 0:
                dp[i, j] = i
            elif tensor1[i - 1] == tensor2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(
                    dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]  # Delete  # Insert
                )  # Substitute

    return dp[m, n]


def is_vis_encoder_real(a):
    from transformers import CLIPVisionModel

    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    convert_weights_to_fp16(model)
    add_prefix = lambda m: {f"model.{k}": v for k, v in m.state_dict().items()}
    b = add_prefix(model)

    out = [(k, torch.allclose(a[k], b[k])) for k in a.keys() if k in a and k in b]
    out = [k for k, _ in filter(lambda i: not i[1], out)]
    diffs = [(k, torch.sum(a[k] - b[k])) for k in out]

    return diffs


def apply_fn_in_batches(tensor, fn, max_batch_size, args):
    out = list(
        map(
            lambda j: fn(tensor[j : j + max_batch_size], *args),
            range(0, tensor.shape[0], max_batch_size),
        )
    )
    out = torch.cat(out)
    return out


def pad_along_dim(list_of_tensors, max_len, dim=0):
    shape = list_of_tensors[0].shape
    pad_num_zeros = 2 * (len(shape) - dim) - 1

    return torch.stack(
        [
            F.pad(t, (0,) * pad_num_zeros + (max_len - t.shape[0],))
            for t in list_of_tensors
        ]
    )
