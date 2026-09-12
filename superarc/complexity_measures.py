"""Complexity of the test sequences themselves -- the bottom panel of Figure 1.

The published Figure 1 is three separately produced images that the typesetter
stacked. Only the top one was ever saved by code:

    top     Success Rate by Model - Simple climbers          22_..._experiments.ipynb
    middle  Success Rate by Model - Random Binary Sequences  22_..._experiments.ipynb
    bottom  BDM / Shannon / zip / lzw over complexity        25_BDM.ipynb

This module owns the bottom panel. It matters beyond the figure: it is the
evidence that the four complexity tiers really are ordered by *algorithmic*
complexity, and that the ordering is agreed on by BDM, Shannon entropy and two
ordinary compressors.

Two things about the original that are preserved deliberately:

* **Only three of the four tiers are plotted, and the third point is tier 4.**
  The pool was generated with the two hardest tiers in the opposite order to the
  one the paper reports, so tiers 3 and 4 are swapped and the new tier 4 dropped
  -- the same convention ``superarc.timeseries`` applies.
* **BDM is computed with the default partition**, not ``PartitionRecursive``.
  ``superarc.table1`` uses the recursive partition; this panel does not. That is
  how the published numbers were produced, so it stays.

The published panel's values were typed into the plotting cell as literals rather
than taken from the cells above it. They are recomputed here, and
``tests/test_complexity_measures.py`` checks them against what was printed. All
but one agree to the digits shown; the third BDM point was transcribed as
``549.678`` where the computation gives ``549.687`` -- two digits transposed. The
difference is far below one pixel on an axis spanning 470-550, so the panel is
unchanged, but the figure no longer depends on hand-copied numbers.
"""

from __future__ import annotations

import base64
import lzma
import os
import zlib
from pathlib import Path

import numpy as np
from pybdm import BDM

from . import REPO_ROOT

# Tier 3 and tier 4 are swapped, then the new tier 4 is dropped -- see the module
# docstring. What remains, in order, is the sequence pool for complexity 1, 2, 3.
PUBLISHED_TIERS = (0, 1, 3)

def sequences_definitons():
    l1 = [
        "1, 2, 3, 4, 5, 6, 7, 8, 9",
        "0, 2, 4, 6, 8, 10, 12",
        "1, 3, 5, 7, 9, 11, 13",
        "101, 103, 105, 107, 109",
        "2, 3, 5, 7, 11, 13, 17, 19, 23",
        "10, 20 ,30 ,40 ,50 ,60",
        "4, 8, 12, 16, 20, 24, 28, 32",
        "3, 6, 9, 12, 15, 18, 21",
        "1, 2, 3",
        "1, 2, 3, 4, 5",
        "2, 4, 6, 8, 10, 12, 14, 16, 18, 20", 
        "3, 6, 9, 12, 15, 18, 21, 24, 27, 30",
        "4, 8, 12, 16, 20, 24, 28, 32, 36, 40",
        "5, 10, 15, 20, 25, 30, 35, 40, 45, 50",
        "6, 12, 18, 24, 30, 36, 42, 48, 54, 60",
        "7, 14, 21, 28, 35, 42, 49, 56, 63, 70",
        "8, 16, 24, 32, 40, 48, 56, 64, 72, 80",
        "9, 18, 27, 36, 45, 54, 63, 72, 81, 90",
        "10, 20, 30, 40, 50, 60, 70, 80, 90, 100",
        "1, 3, 5, 7, 9, 11, 13, 15, 17, 19",
        "2, 4, 6, 8, 10, 12, 14, 16, 18, 20",
        "11, 12, 13, 14, 15, 16, 17, 18, 19, 20",
        "21, 22, 23, 24, 25, 26, 27, 28, 29, 30",
        "31, 32, 33, 34, 35, 36, 37, 38, 39, 40",
        "41, 42, 43, 44, 45, 46, 47, 48, 49, 50",
        "51, 52, 53, 54, 55, 56, 57, 58, 59, 60",
        "61, 62, 63, 64, 65, 66, 67, 68, 69, 70",
        "71, 72, 73, 74, 75, 76, 77, 78, 79, 80",
        "81, 82, 83, 84, 85, 86, 87, 88, 89, 90",
        "91, 92, 93, 94, 95, 96, 97, 98, 99, 100",
        "101, 102, 103, 104, 105, 106, 107, 108, 109, 110",
        "111, 112, 113, 114, 115, 116, 117, 118, 119, 120",
        "121, 122, 123, 124, 125, 126, 127, 128, 129, 130",
        "131, 132, 133, 134, 135, 136, 137, 138, 139, 140",
        "141, 142, 143, 144, 145, 146, 147, 148, 149, 150",
        "151, 152, 153, 154, 155, 156, 157, 158, 159, 160",
        "161, 162, 163, 164, 165, 166, 167, 168, 169, 170",
        "171, 172, 173, 174, 175, 176, 177, 178, 179, 180",
        "181, 182, 183, 184, 185, 186, 187, 188, 189, 190",
        "191, 192, 193, 194, 195, 196, 197, 198, 199, 200",
    ]
    
    l2 = [
        "0, 1, 1, 2, 3, 5, 8, 13",
        "0, 6, 9, 3, 1, 4, 7",
        "0, 0, 0, 0, 0, 0",
        "0, 1, 2, 4, 8, 16",
        "1, 2, 4, 7, 11, 16, 22, 29",
        "1, 1, 1, 2, 3, 4, 6, 9",
        "30, 42, 66, 70, 78, 102",
        "0, 1, 4, 9, 16, 25, 36",
        "0, 1, 8, 27, 64, 125, 216, 343, 512",
        "4, 6, 9, 10, 14, 15, 21, 22, 25, 26",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 1, 2, 3, 5, 8, 13, 21, 34, 55",
        "1, 2, 4, 8, 16, 32, 64, 128, 256, 512",
        "1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683",
        "1, 4, 9, 16, 25, 36, 49, 64, 81, 100",
        "1, 8, 27, 64, 125, 216, 343, 512, 729, 1000",
        "1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880",
        "1, 3, 6, 10, 15, 21, 28, 36, 45, 55",
        "2, 1, 3, 4, 7, 11, 18, 29, 47, 76",
        "0, 1, 2, 5, 12, 29, 70, 169, 408, 985",
        "1, 4, 27, 256, 3125, 46656, 823543, 16777216, 387420489, 10000000000",
        "1, 2, 6, 20, 70, 252, 924, 3432, 12870, 48620",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "4, 6, 9, 10, 14, 15, 21, 22, 25, 26",
        "1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010",
        "0, 1, 81, 512, 2401, 4913, 5832, 17576, 19683, 234256",
        "1, 2, 145, 40585",
        "2, 5, 12, 20, 29, 39, 50, 62, 75, 89",
        "1, 8, 10, 18, 19, 100, 101, 108, 109, 110",
        "3, 7, 31, 127, 2047, 8191, 131071, 524287, 8388607, 536870911",
        "1, 2, 4, 8, 16, 23, 28, 38, 58, 89",
        "1, 2, 4, 8, 15, 26, 42, 64, 93, 129",
        "1, 5, 12, 22, 35, 51, 70, 92, 117, 145",
        "0, 1, 1, 2, 1, 2, 2, 3, 1, 3",
        "1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 11, 21, 1211, 111221",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 2, 4, 8, 16, 32, 64, 128, 256, 512",
        "1, 3, 7, 15, 31, 63, 127, 255, 511, 1023",
    ]
    
    l3 = [
        "3, 1, 4 ,1 ,5 ,9 ,2 ,6 , 5",
        "1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10",
        "1, 6, 1, 8, 0, 3, 3, 9, 8, 8, 7",
        "1, 1, 4, 9, 4, 2, 0, 4, 4, 8, 5",
        "2, 0, 9, 4, 5, 5, 1, 4, 8, 1, 5",
        "1, 11, 21, 1211, 111221, 312211, 13112221, 1113213211",
        "2, 3, 5, 11, 23, 29, 41, 53, 83, 89, 113, 131",
        "1, 7, 10, 13, 19, 23, 28, 31, 32, 44",
        "0, 1, 4, 10, 20, 35, 56, 84, 120, 165",
        "1, 2, 5, 13, 29, 34, 89, 169, 194",
        "1, 4, 9, 16, 25, 36, 49, 64, 81, 100",
        "1, 8, 27, 64, 125, 216, 343, 512, 729, 1000",
        "2, 1, 2, 1, 2, 2, 1, 2, 2, 2",
        "1, 2, 2, 3, 3, 4, 4, 4, 5, 5",
        "2, 4, 8, 16, 32, 64, 128, 256, 512, 1024",
        "2, 5, 12, 20, 29, 39, 50, 62, 75, 89",
        "7, 11, 13, 17, 19, 23, 29, 31, 37, 41",
        "1, 11, 111, 1111, 11111, 111111, 1111111, 11111111, 111111111, 1111111111",
        "31, 331, 3331, 33331, 333331, 3333331, 33333331, 333333331",
        "1, 3, 7, 13, 21, 31, 43, 57, 73, 91",
        "6, 28, 496, 8128, 130816, 2096128",
        "0, 0, 2, 3, 5, 7, 11, 15, 22, 30",
        "3, 37, 67, 101, 137, 199, 269, 307, 353, 389",
        "1, 2, 145, 40585",
        "19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203",
        "2, 3, 7, 42, 1806, 3263442",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 8, 10, 18, 19, 100, 101, 108, 109, 110",
        "0, 1, 1, 2, 1, 2, 2, 3, 1, 3",
        "0, 1, 1, 2, 2, 4, 2, 6, 4, 6",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 4, 2, 1, 3, 5, 9, 2, 6, 5",
        "1, 7, 3, 2, 0, 5, 0, 7, 9, 8",
        "1, 0, 1, 2, 3, 1, 6, 11",
        "1, 1, 1, 3, 7, 15, 31, 63",
        "3, 5, 11, 17, 31, 41, 59, 67",
        "1, 2, 3, 4, 6, 8, 11, 13",
        "1, 0, 1, 1, 2, 3, 6, 11",
        "1093, 3511",
        "5, 13, 563",
    ]
    l4 = [
        "2, 6, 8, 5, 0, 3, 7, 4, 5, 6, 7, 8, 9, 0",
        "62, 76, 58, 75, 90, 33, 77, 94, 65, 56, 27",
        "6, 7, 27, 58, 544, 695, 578, 4726, 8878, 12472, 89098",
        "29, 57, 68, 120, 134, 140, 173, 197, 283, 313",
        "24, 26, 36, 40, 184, 226, 244, 384, 391, 423",
        "90, 203, 212, 235, 270, 324, 342, 352, 371, 417",
        "20, 48, 95, 234, 282, 296, 352, 402, 428, 481",
        "62, 98, 130, 154, 290, 315, 324, 385, 408, 447",
        "2, 42, 66, 102, 153, 195, 201, 252, 306, 396",
        "128, 151, 153, 217, 224, 332, 382, 400, 450, 478",
        "26, 50, 114, 148, 160, 170, 274, 347, 432, 497",
        "48, 94, 176, 177, 219, 276, 282, 283, 459, 488",
        "139, 252, 272, 281, 304, 361, 370, 415, 438, 500",
        "15, 95, 115, 195, 240, 318, 326, 350, 432, 450",
        "134, 224, 293, 378, 379, 395, 434, 451, 482, 496",
        "23, 93, 142, 145, 245, 266, 296, 317, 428, 495",
        "18, 39, 71, 194, 197, 219, 263, 270, 416, 473",
        "9, 84, 144, 170, 325, 393, 401, 405, 435, 497",
        "26, 40, 202, 267, 282, 340, 359, 408, 410, 495",
        "34, 92, 164, 165, 209, 296, 414, 456, 467, 494",
        "16, 119, 121, 123, 135, 139, 285, 311, 409, 412",
        "8, 11, 12, 103, 116, 196, 247, 254, 389, 427",
        "12, 36, 96, 119, 171, 213, 221, 232, 363, 451",
        "38, 91, 142, 197, 215, 313, 316, 319, 423, 466",
        "7, 42, 147, 201, 213, 248, 310, 332, 436, 479",
        "27, 101, 105, 164, 245, 290, 304, 441, 449, 490",
        "4, 11, 29, 106, 214, 283, 296, 298, 360, 497",
        "72, 106, 139, 165, 171, 192, 199, 429, 453, 477",
        "187, 218, 260, 295, 301, 314, 379, 410, 452, 469",
        "29, 63, 95, 140, 150, 190, 221, 437, 482, 491",
        "3, 11, 84, 144, 156, 177, 188, 199, 229, 284",
        "26, 94, 98, 137, 176, 301, 323, 330, 372, 444",
        "39, 81, 88, 210, 215, 378, 416, 430, 439, 490",

    ]
    
    return l1,l2, l3, l4


def compress_text(text):
    # Apply LZW compression using lzma (LZMA is a variant of LZW)
    lzw_compressed = lzma.compress(text.encode('utf-8'))
    
    # Apply ZIP compression using zlib
    zip_compressed = zlib.compress(text.encode('utf-8'))
    
    # Encode the compressed data to Base64
    lzw_compressed_base64 = base64.b64encode(lzw_compressed).decode('utf-8')
    zip_compressed_base64 = base64.b64encode(zip_compressed).decode('utf-8')
    
    return lzw_compressed_base64, zip_compressed_base64


def ascii_to_binary_list(text):
    binary_list = []
    for char in text:
        # Convert each character to its binary representation
        binary_representation = format(ord(char), '08b')
        # Extend the binary_list with the numerical digits of the binary representation
        binary_list.extend([int(bit) for bit in binary_representation])
    return binary_list


def list_of_strings_to_binary_lists(strings_list):
    binary_lists = [ascii_to_binary_list(string) for string in strings_list]
    return binary_lists


def list_of_strings_to_compressed(strings_list):
    lzw_compressed_list = []
    zip_compressed_list = []
    
    for string in strings_list:
        lzw_compressed, zip_compressed = compress_text(string)
        lzw_compressed_list.append(lzw_compressed)
        zip_compressed_list.append(zip_compressed)
    
    return lzw_compressed_list, zip_compressed_list


def normalize_multiple_binary_sequences(binary_seq_sets):
    # a) Compute average length for each set
    avg_lengths = [round(np.mean([len(seq) for seq in binary_seq])) for binary_seq in binary_seq_sets]
    
    # b) Select the lower calculated average
    min_avg_length = min(avg_lengths)
    
    # c) Modify each sequence in each set to the lower average length
    normalized_sets = [
        [np.array(seq[:min_avg_length], dtype=int) for seq in binary_seq if len(seq) >= min_avg_length]
        for binary_seq in binary_seq_sets
    ]
    
    return min_avg_length, normalized_sets


def compute_avg_bdm(norm_bin_sequences_list):
    bdm = BDM(ndim=1)
    bdm_values = [bdm.bdm(seq) for seq in norm_bin_sequences_list]
    shannon_values = [bdm.ent(seq) for seq in norm_bin_sequences_list]
    return np.mean(bdm_values), np.mean(shannon_values)


def list_of_lists_to_np_arrays(list_of_lists):
    return [np.array(sublist) for sublist in list_of_lists]


def transform_to_np_arrays(list_of_lists_of_lists):
    return [list_of_lists_to_np_arrays(sublist) for sublist in list_of_lists_of_lists]


def process_binary_sequences(binary_seq_sets, normalize=True):
    # Normalize all sets to the minimum average length
    if not normalize:
        min_avg_length = None
        normalized_bin_sets = transform_to_np_arrays(binary_seq_sets)
    else:
        min_avg_length, normalized_bin_sets = normalize_multiple_binary_sequences(binary_seq_sets)
    
    # d) Calculate avg bdm and avg shannon for each normalized set
    avg_bdm_values, avg_shannon_values = zip(*(compute_avg_bdm(norm_set) for norm_set in normalized_bin_sets))
    
    return min_avg_length, list(avg_bdm_values), list(avg_shannon_values)



def average_length_of_strings(strings_list):
    total_length = sum(len(s) for s in strings_list)
    return total_length / len(strings_list) if strings_list else 0


def measures() -> dict[str, list[float]]:
    """The four complexity measures over the published tiers.

    ``BDM`` and ``Shannon`` are computed on the raw sequences, normalised to a
    common length so that the tiers are comparable. ``zip`` and ``lzw`` are the
    mean length of the compressed, base64-encoded sequence -- not normalised,
    because compressed length is the measure.
    """
    pools = sequences_definitons()

    _, bdm_values, shannon_values = process_binary_sequences(
        [list_of_strings_to_binary_lists(pool) for pool in pools]
    )

    zip_lengths, lzw_lengths = [], []
    for pool in pools:
        lzw_compressed, zip_compressed = list_of_strings_to_compressed(pool)
        lzw_lengths.append(average_length_of_strings(lzw_compressed))
        zip_lengths.append(average_length_of_strings(zip_compressed))

    pick = lambda values: [float(values[i]) for i in PUBLISHED_TIERS]  # noqa: E731
    return {
        "BDM": pick(bdm_values),
        "Shannon": pick(shannon_values),
        "zip": pick(zip_lengths),
        "lzw": pick(lzw_lengths),
    }


PANEL_COLOURS = {"BDM": "blue", "Shannon": "green", "zip": "red", "lzw": "purple"}


def output_dir() -> Path:
    override = os.environ.get("SUPERARC_PLOTS_DIR")
    return Path(override) if override else REPO_ROOT / "plots" / "highResolution"


def plot(values: dict[str, list[float]] | None = None, out_dir: Path | None = None) -> list[Path]:
    """Draw and save the bottom panel of Figure 1."""
    import matplotlib.pyplot as plt

    values = values or measures()
    out_dir = out_dir or output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (name, series) in zip(axes.flat, values.items()):
        ax.plot([1, 2, 3], series, label=name, color=PANEL_COLOURS[name], marker="o")
        ax.set_title(name, fontsize=15)
        ax.set_xlabel("Index", fontsize=15)
        ax.set_ylabel("Values", fontsize=15)
        ax.set_xticks([1, 2, 3])
        ax.legend(fontsize=15)
    fig.tight_layout()

    written = []
    for suffix in (".pdf", ".svg", ".png"):
        path = out_dir / ("figure01-bottom" + suffix)
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> int:
    values = measures()
    for name, series in values.items():
        print(f"{name:>8s}  " + "  ".join(f"{v:10.4f}" for v in series))
    for path in plot(values):
        if path.suffix == ".png":
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
