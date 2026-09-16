"""The polyglot programming-language experiment -- Supplementary Figures 2 to 4.

GPT-4 was asked for the same sequences in seven languages (Mathematica, Matlab,
Python, ArnoldC, JavaScript, C++, R) at four complexity tiers and several
temperatures. Five panels were printed, across three supplementary figures:

    figure11-top      Correct executions and prints, by language and complexity
    figure11_bottom   Which sequences each language got right (supervenn)
    figure12-top      Correct prints, heatmap by language and complexity
    figure12-bottom   No-compression percentage vs complexity, log scale
    figure13          No-compression percentage by complexity, per language,
                      with temperature variation

Why this module exists
----------------------
These five had **no runnable producer**. ``24_PLOTS_paper.ipynb`` draws them, but
it read one input through an absolute path on another machine
(``/Users/beto/Documents/Projects/medLLM/normalized_compressed_ArnoldC.csv``), so
it could not run here -- and the same absolute path is in the lab repository's
copy (``AlgoDynLab/SuperintelligenceTest``, ``05_PLOTS_paper.ipynb``), which is
otherwise the identical notebook. The file it names has been in this repository
all along. That one line was the whole blocker, and these were the last published
figures with nothing checking them.

Ported 2026-09-15 and verified against the committed artifacts before anything
was rewritten:

    figure11-top      0.0000%
    figure12-top      0.0000%
    figure12-bottom   0.0000%
    figure13          0.0000%
    figure11_bottom   19.2% with PYTHONHASHSEED=0, and unstable without it
                      -- see below

``figure11_bottom`` is a ``supervenn`` plot with ``sets_ordering='minimize gaps'``.
**Its data is unchanged** -- all seven set sizes match the published figure
exactly (Mathematica 40, Python 52, Java 69, Matlab 72, R 59, Cpp 45,
ArnoldC 12), and the rows are in the same order. Only the column packing differs.

The packing is not a library-version difference, which is what a first pass here
claimed on the strength of two runs that agreed at 0.0000%. Those two runs shared
a process, and so shared a hash seed, which is exactly the thing that varies.
Across *separate* processes the same code disagrees with itself by 15.7%, 24.8%
and 25.7% on three runs.

The cause is that the panel's inputs are ``set`` objects holding sequence
*strings*. Python randomises string hashing per process, so set iteration order
changes from run to run, and ``sets_ordering='minimize gaps'`` is a heuristic over
that order. With ``PYTHONHASHSEED=0`` two separate processes agree at 0.0000%.

So this panel joins the bootstrap tiers and Supplementary Figure 1: the published
image depends on a draw nobody recorded -- here a hash seed rather than an RNG
seed -- and cannot be recovered. Seeded, it is stable at 19.2% from print, and the
numbers underneath it are identical. ``superarc.parity`` sets ``PYTHONHASHSEED=0``
for every producer so this is held to run-to-run determinism.

Edits to the moved bodies
-------------------------
One, declared here and nowhere else: the two loaders called
``pd.read_csv("<name>.csv")`` with a bare relative name, so they resolved against
the current working directory and worked only when run from the repository root.
They broke the moment a notebook in ``notebooks/`` imported them, and nothing
else in ``superarc`` resolves inputs that way. They now read
``REPO_ROOT / "<name>.csv"``. The files read are unchanged, and the four exact
panels still reproduce at 0.0000%.

Ownership
---------
The loaders and the per-language summary chain were in ``processing_answers.py``.
They are here now, character for character, and that module re-exports them --
the same direction already taken for ``levenshtein`` and friends in
:mod:`superarc.timeseries`. Nothing in ``superarc`` imports the historical
module.

The complexity swap, for the third time
---------------------------------------
Cell 1 of the notebook does::

    total_df['complexity'] = total_df['complexity'].replace({4: 3, 3: 4})

in place, so running the cell twice swaps the two hardest tiers back. This is the
third occurrence of that exact hazard in this repository (see
:func:`superarc.timeseries.average_by_complexity`, and cell 53 of
``22_Timeseries_LLM_experiments.ipynb``). :func:`swap_hardest_tiers` returns a
copy.

**The cell order is load-bearing and is preserved in :func:`produce`.** Cell 9
*reassigns* ``total_df = concat_all_results()``, discarding the swap, so the
first three panels are drawn from swapped data and the last two from unswapped
data. That is not tidy, but it is what produced the published figures, and
changing it moves them.

One side effect is deliberately not preserved: ``analyze_all_languages`` passed
``save_language_name`` for every language, so running it rewrote the seven
committed ``normalized_compressed_*.csv`` in the repository root. They regenerate
byte-identically, but a producer that writes into the source tree is exactly what
``plots_dir()`` exists to prevent, so :func:`analysis` defaults to not writing
them.
"""

from __future__ import annotations

import binascii
import copy
import zlib
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from . import REPO_ROOT, plots_dir

LANGUAGES = ("Mathematica", "Matlab", "Python", "ArnoldC", "JavaScript", "C++", "R")

# The supervenn panel's set labels and the language each denotes, in the order
# the published figure stacks them.
SUPERVENN_LABELS = ("Mathematica", "R", "Matlab", "Python", "Cpp", "Java", "ArnoldC")
SUPERVENN_LANGUAGES = ("Mathematica", "R", "Matlab", "Python", "C++", "JavaScript",
                       "ArnoldC")


def swap_hardest_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    """Exchange complexity 3 and 4, on a copy.

    The notebook does this in place and is therefore not idempotent; see the
    module docstring.
    """
    out = frame.copy()
    out["complexity"] = out["complexity"].replace({4: 3, 3: 4})
    return out



# ----------------------------------------------------------------------------
# Moved verbatim from processing_answers.py, which now re-exports them.
# ----------------------------------------------------------------------------

def concat_all_results(verbose=False):
    """Load all datasets

    Args:
        verbose (bool, optional): if you wanna see some details. Defaults to False.

    Returns:
        df: a dataframe with all datasets concatenated
    """
    mathe_df = pd.read_csv(REPO_ROOT / "automatic_mathematica_no_comments.csv")
    matlab_df = pd.read_csv(REPO_ROOT / "automatic_matlab_no_comments.csv")
    py_df = pd.read_csv(REPO_ROOT / "automatic_python_no_comments.csv")
    arnoldC_df = pd.read_csv(REPO_ROOT / "automatic_arnoldC.csv")
    js_df = pd.read_csv(REPO_ROOT / "javascript_evaluation_no_comments.csv")  # this was not labeled with print_code automaticallyF
    cpp_df = pd.read_csv(REPO_ROOT / "Cpp_evaluation_no_comments.csv")
    r_df = pd.read_csv(REPO_ROOT / "R_evaluation_no_comments.csv")

    total_df = pd.concat([mathe_df, matlab_df, py_df, arnoldC_df, js_df, cpp_df, r_df])
    if verbose:
        print("Length: ", str(len(total_df)))
        print("Columns: ", list(total_df.columns))
        print("Languages: ", list(total_df["language"].unique()))

    return total_df


def concat_all_compressed_files(verbose=False):
    """Load all datasets

    Args:
        verbose (bool, optional): if you wanna see some details. Defaults to False.

    Returns:
        df: a dataframe with all datasets concatenated
    """
    mathe_df = pd.read_csv(REPO_ROOT / "normalized_compressed_Mathematica.csv")
    matlab_df = pd.read_csv(REPO_ROOT / "normalized_compressed_Matlab.csv")
    py_df = pd.read_csv(REPO_ROOT / "normalized_compressed_Python.csv")
    arnoldC_df = pd.read_csv(REPO_ROOT / "normalized_compressed_ArnoldC.csv")
    js_df = pd.read_csv(REPO_ROOT / "normalized_compressed_JavaScript.csv")  # this was not labeled with print_code automaticallyF
    cpp_df = pd.read_csv(REPO_ROOT / "normalized_compressed_C++.csv")
    r_df = pd.read_csv(REPO_ROOT / "normalized_compressed_R.csv")

    total_df = pd.concat([mathe_df, matlab_df, py_df, arnoldC_df, js_df, cpp_df, r_df])
    if verbose:
        print("Length: ", str(len(total_df)))
        print("Columns: ", list(total_df.columns))
        print("Languages: ", list(total_df["language"].unique()))

    return total_df


def str2hex(s):
    return binascii.hexlify(s)


def hex2str(h):
    return binascii.unhexlify(h)


def no_compression_units(code_string, target_seq_str):
    numbers_list = target_seq_str.split(",")
    for i in range(len(numbers_list)):
        numbers_list[i] = numbers_list[i].replace(" ", "")
    # print(numbers_list)
    no_compression_measure = 0
    for n in numbers_list:
        if n in code_string:
            no_compression_measure += 1
    no_compression_percentage = (no_compression_measure * 100) / len(numbers_list)
    return no_compression_percentage


def compress_line(line):
    """returns a line coded

    Args:
        line (str): text to be coded

    Returns:
        str: coded string
    """
    line_encoded_utf8 = line.encode("utf-8")
    # print(f"{line_encoded_utf8=}")

    line_compressed = zlib.compress(line_encoded_utf8)
    # print(f"{line_compressed=}")

    compressed_hex = str2hex(line_compressed)
    # print(f"{compressed_hex=}")

    compressed_hex_str = compressed_hex.decode("utf-8")
    # print(f"{compressed_hex_str=}")
    return compressed_hex_str


def compress_answers(one_language_df):
    """Given a dataset of a single language, compress/code answers and measure its length
    if the coded one

    Args:
        one_language_df (df): target dataframe

    Returns:
        df: a dataframe agmented with the compressed answers and its length
    """
    # compress the original answers
    compressed_df = copy.deepcopy(one_language_df)
    cols = list(compressed_df.columns)
    compressed_df["compressed_answer"] = compressed_df["GPT4_answer"].apply(
        compress_line
    )
    compressed_df["compressed_len"] = compressed_df["compressed_answer"].apply(len)

    # if dataframe is normalized, also compress the normalized answers
    if "normalized_answer" in cols:
        compressed_df["compressed_norm_answer"] = compressed_df[
            "normalized_answer"
        ].apply(compress_line)
        compressed_df["compressed_norm_len"] = compressed_df[
            "compressed_norm_answer"
        ].apply(len)

    # compress original sequences with no code given by chatGPT
    compressed_df["compressed_sequence"] = compressed_df["sequence"].apply(
        compress_line
    )
    compressed_df["compressed_sequence_len"] = compressed_df[
        "compressed_sequence"
    ].apply(len)
    return compressed_df


def normalize_prints_and_no_compression_measure(language_df):
    """Normalize answers taking definiton out the original sequence from the code
    of the answer

    Args:
        language_df (df): a language dataframe

    Returns:
        df: dataframe with normalized answers
    """
    copy_df = copy.deepcopy(language_df)

    sequences = list(copy_df["sequence"].values)
    original_answers = list(copy_df["GPT4_answer"].values)
    original_lengths = list(copy_df["a_length"].values)
    print_cases = list(copy_df["print_code"].values)

    normalized_answer = copy.deepcopy(original_answers)
    normalized_length = copy.deepcopy(original_lengths)
    no_compression_percentage_list = [0] * len(original_answers)

    for i in range(len(print_cases)):
        is_print = print_cases[i] == 1
        if is_print:
            seq = sequences[i]
            n_answer = original_answers[i].replace(seq, "")
            n_length = len(n_answer)
            normalized_length[i] = n_length
            normalized_answer[i] = n_answer

    copy_df["normalized_answer"] = normalized_answer
    copy_df["normalized_length"] = normalized_length

    # compute no compression percentage
    for i in range(len(sequences)):
        no_compression_percentage = no_compression_units(
            original_answers[i], sequences[i]
        )
        no_compression_percentage_list[i] = no_compression_percentage

    copy_df["no_compression_percentage"] = no_compression_percentage_list
    return copy_df


def compute_mean_column(df, column_name):
    return df.loc[:, column_name].mean()


def describe_df(one_language_df, complexity, verbose=False):

    one_complexity_df = copy.deepcopy(
        one_language_df[one_language_df["complexity"] == complexity]
    )

    # print("     applying complexity filtering: ")
    # print("     looking for complexity: ", str(complexity))
    # print(
    #     "      after appply filter: ",
    #     str(list(one_complexity_df["complexity"].unique())),
    # )
    cols = list(one_complexity_df.columns)
    avg_answer_len = None  # original answer length
    norm_avg_answer_len = None  # normalized answer length
    compressed_avg_len = None  # original compressed answer length
    compressed_norm_avg_len = None  # normalied compressed answer length
    compressed_seq_avg_len = None  # pure compressed seq
    avg_no_compression_percentage = None

    # if dataframe has been normalized and compressed
    if ("normalized_answer" in cols) and ("compressed_answer" in cols):
        avg_answer_len = one_complexity_df.loc[:, "a_length"].mean()
        norm_avg_answer_len = one_complexity_df.loc[:, "normalized_length"].mean()
        compressed_avg_len = one_complexity_df.loc[:, "compressed_len"].mean()
        compressed_norm_avg_len = one_complexity_df.loc[:, "compressed_norm_len"].mean()
        compressed_seq_avg_len = one_complexity_df.loc[
            :, "compressed_sequence_len"
        ].mean()
        avg_no_compression_percentage = one_complexity_df.loc[
            :, "no_compression_percentage"
        ].mean()

        if verbose:
            print(
                "[norm_avg_lenght, avg_length, compress_avg_len] for complexity {}:[{}, {}, {}, {}, {}]".format(
                    complexity,
                    "%.2f" % (norm_avg_answer_len),
                    "%.2f" % (avg_answer_len),
                    "%.2f" % (compressed_avg_len),
                    "%.2f" % (compressed_norm_avg_len),
                    "%.2f" % (compressed_seq_avg_len),
                )
            )

    elif not ("normalized_answer" in cols) and not ("compressed_answer" in cols):
        avg_answer_len = one_complexity_df.loc[:, "a_length"].mean()
        if verbose:
            print(
                "avg_length for complexity {}: {}".format(
                    complexity, "%.2f" % (avg_answer_len)
                )
            )

    elif ("normalized_answer" in cols) and not ("compressed_answer" in cols):
        avg_answer_len = one_complexity_df.loc[:, "a_length"].mean()
        norm_avg_answer_len = one_complexity_df.loc[:, "normalized_length"].mean()

        if verbose:
            print(
                "[norm_avg_lenght, avg_length] for complexity {}: [{}, {}]".format(
                    complexity,
                    "%.2f" % (norm_avg_answer_len),
                    "%.2f" % (avg_answer_len),
                )
            )

    elif not ("normalized_answer" in cols) and ("compressed_answer" in cols):
        avg_answer_len = one_complexity_df.loc[:, "a_length"].mean()
        compressed_avg_len = one_complexity_df.loc[:, "compressed_len"].mean()
        compressed_seq_avg_len = one_complexity_df.loc[
            :, "compressed_sequence_len"
        ].mean()

        if verbose:
            print(
                "[norm_avg_lenght, avg_length, compress_avg_len] for complexity {}:[{}, {}, {}]".format(
                    complexity,
                    "%.2f" % (avg_answer_len),
                    "%.2f" % (compressed_avg_len),
                    "%.2f" % (compressed_seq_avg_len),
                )
            )

    return (
        avg_answer_len,  # original length
        norm_avg_answer_len,  # normalized length
        compressed_avg_len,  # compressed original legth
        compressed_norm_avg_len,  # compressed normalized length
        compressed_seq_avg_len,  # compressed pure sequences length
        avg_no_compression_percentage,  # avg no compression percentage
    )


def summarize_df_by_filter(
    one_language_df,
    complexity,
    only_print=False,
    only_correct_exe=False,
    correct_exe_filter=False,
    print_code_filter=False,
    verbose=False,
):
    aux = copy.deepcopy(one_language_df)
    len_before_filter = len(aux)
    # no filter case
    if (
        (correct_exe_filter is False)
        and (print_code_filter is False)
        and (only_print is False)
        and (only_correct_exe is False)
    ):
        language = aux["language"].unique()[0]
        if language == "ArnoldC":
            filtered = copy.deepcopy(aux[aux["correct_execution"] == True])
        else:
            filtered = copy.deepcopy(aux)
    # only print filter applied
    elif only_print is True:
        filtered = copy.deepcopy(aux[aux["print_code"] == 1])
    # only correct filter applied
    elif only_correct_exe is True:
        filtered = copy.deepcopy(aux[aux["correct_execution"] == True])
    # correct that are not prints
    elif (correct_exe_filter is True) and (print_code_filter is False):
        filtered = copy.deepcopy(
            aux[(aux["correct_execution"] == True) & (aux["print_code"] == 0)]
        )
    # incorrect prints
    elif (correct_exe_filter is False) and (print_code_filter is True):
        filtered = copy.deepcopy(
            aux[(aux["correct_execution"] == False) & (aux["print_code"] == 1)]
        )
    # print and correct filters applied
    elif (correct_exe_filter is True) and (print_code_filter is True):
        filtered = copy.deepcopy(aux[aux["correct_execution"] == True])
        filtered = copy.deepcopy(filtered[filtered["print_code"] == 1])

    # print("*** Applying FILTERS ***")
    # print("LOOKING FOR complexity: ", str(complexity))
    # print("filters [c, p]:", str(correct_exe_filter), str(print_code_filter))
    # print("contained complexity:", str(list(filtered["complexity"].unique())))
    len_after_filter = len(filtered)
    if len_after_filter == 0:
        len_after_filter = 1
    coincidence_percentage = (len_after_filter * 100) / len_before_filter

    (
        avg_answer_len,
        norm_avg_answer_len,
        compressed_avg_len,
        compressed_norm_avg_len,
        compressed_seq_avg_len,
        avg_no_compression_percentage,
    ) = describe_df(one_language_df=filtered, complexity=complexity, verbose=verbose)

    return (
        coincidence_percentage,
        avg_answer_len,
        norm_avg_answer_len,
        compressed_avg_len,
        compressed_norm_avg_len,
        compressed_seq_avg_len,
        avg_no_compression_percentage,
        filtered,
    )


def range1(start, end):
    return range(start, end + 1)


def create_language_summary_df(rough_language_df, verbose=False, save_language_name=""):
    normalize_df = normalize_prints_and_no_compression_measure(rough_language_df)
    normalize_df = compress_answers(normalize_df)
    if save_language_name != "":
        normalize_df.to_csv(
            "normalized_compressed_{}.csv".format(save_language_name), index=False
        )

    # Original (nf = no filter)
    nf_percentage_arr = [0] * 4
    nf_avg_len_arr = [0] * 4
    nf_norm_avg_len_arr = [0] * 4
    nf_compr_avg_len_arr = [0] * 4
    nf_compr_norm_avg_len_arr = [0] * 4
    nf_compr_seq_avg_len_arr = [0] * 4
    nf_avg_no_compression_percentage_arr = [0] * 4
    # only print filter
    p_percentage_arr = [0] * 4
    p_avg_len_arr = [0] * 4
    p_norm_avg_len_arr = [0] * 4
    p_compr_avg_len_arr = [0] * 4
    p_compr_norm_avg_len_arr = [0] * 4
    p_compr_seq_avg_len_arr = [0] * 4
    p_avg_no_compression_percentage_arr = [0] * 4
    # only correct filter
    c_percentage_arr = [0] * 4
    c_avg_len_arr = [0] * 4
    c_norm_avg_len_arr = [0] * 4
    c_compr_avg_len_arr = [0] * 4
    c_compr_norm_avg_len_arr = [0] * 4
    c_compr_seq_avg_len_arr = [0] * 4
    c_avg_no_compression_percentage_arr = [0] * 4
    # correct no prints
    cnp_percentage_arr = [0] * 4
    cnp_avg_len_arr = [0] * 4
    cnp_norm_avg_len_arr = [0] * 4
    cnp_compr_avg_len_arr = [0] * 4
    cnp_compr_norm_avg_len_arr = [0] * 4
    cnp_compr_seq_avg_len_arr = [0] * 4
    cnp_avg_no_compression_percentage_arr = [0] * 4
    # incorrect prints
    ip_percentage_arr = [0] * 4
    ip_avg_len_arr = [0] * 4
    ip_norm_avg_len_arr = [0] * 4
    ip_compr_avg_len_arr = [0] * 4
    ip_compr_norm_avg_len_arr = [0] * 4
    ip_compr_seq_avg_len_arr = [0] * 4
    ip_avg_no_compression_percentage_arr = [0] * 4
    # print and correct filter
    p_c_percentage_arr = [0] * 4
    p_c_avg_len_arr = [0] * 4
    p_c_norm_avg_len_arr = [0] * 4
    p_c_compr_avg_len_arr = [0] * 4
    p_c_compr_norm_avg_len_arr = [0] * 4
    p_c_compr_seq_avg_len_arr = [0] * 4
    p_c_avg_no_compression_percentage_arr = [0] * 4

    lan = rough_language_df["language"].unique()[0]
    language_col = [lan, lan, lan, lan]
    complexity_arr = []
    output = pd.DataFrame()

    counter = -1
    for complexity in range1(1, 4):
        counter = counter + 1
        # No filter
        (
            nf_percentage,
            nf_avg_answer_len,
            nf_norm_avg_answer_len,
            nf_compressed_avg_len,
            nf_compressed_norm_avg_len,
            nf_compressed_seq_avg_len,
            nf_no_copression_percentage,
            nf_filtered_df,
        ) = summarize_df_by_filter(normalize_df, complexity, verbose=verbose)

        # only print ("p" prefix) filter applied
        (
            p_percentage,
            p_avg_answer_len,
            p_norm_avg_answer_len,
            p_compressed_avg_len,
            p_compressed_norm_avg_len,
            p_compressed_seq_avg_len,
            p_no_copression_percentage,
            p_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df, complexity, only_print=True, verbose=verbose
        )

        # only correct ("c" prefix) filter applied
        (
            c_percentage,
            c_avg_answer_len,
            c_norm_avg_answer_len,
            c_compressed_avg_len,
            c_compressed_norm_avg_len,
            c_compressed_seq_avg_len,
            c_no_copression_percentage,
            c_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df, complexity, only_correct_exe=True, verbose=verbose
        )

        # Correct that are not print
        (
            cnp_percentage,
            cnp_avg_answer_len,
            cnp_norm_avg_answer_len,
            cnp_compressed_avg_len,
            cnp_compressed_norm_avg_len,
            cnp_compressed_seq_avg_len,
            cnp_no_copression_percentage,
            cnp_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df,
            complexity,
            correct_exe_filter=True,
            print_code_filter=False,
            verbose=verbose,
        )

        # incorrect prints
        (
            ip_percentage,
            ip_avg_answer_len,
            ip_norm_avg_answer_len,
            ip_compressed_avg_len,
            ip_compressed_norm_avg_len,
            ip_compressed_seq_avg_len,
            ip_no_copression_percentage,
            ip_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df,
            complexity,
            correct_exe_filter=False,
            print_code_filter=True,
            verbose=verbose,
        )

        # print and correct filters applied
        (
            p_c_percentage,
            p_c_avg_answer_len,
            p_c_norm_avg_answer_len,
            p_c_compressed_avg_len,
            p_c_compressed_norm_avg_len,
            p_c_compressed_seq_avg_len,
            p_c_no_copression_percentage,
            p_c_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df,
            complexity,
            print_code_filter=True,
            correct_exe_filter=True,
            verbose=verbose,
        )

        # Original (no transformation)
        nf_percentage_arr[counter] = nf_percentage
        nf_avg_len_arr[counter] = nf_avg_answer_len
        nf_norm_avg_len_arr[counter] = nf_norm_avg_answer_len
        nf_compr_avg_len_arr[counter] = nf_compressed_avg_len
        nf_compr_norm_avg_len_arr[counter] = nf_compressed_norm_avg_len
        nf_compr_seq_avg_len_arr[counter] = nf_compressed_seq_avg_len
        nf_avg_no_compression_percentage_arr[counter] = nf_no_copression_percentage
        # Print filter
        p_percentage_arr[counter] = p_percentage
        p_avg_len_arr[counter] = p_avg_answer_len
        p_norm_avg_len_arr[counter] = p_norm_avg_answer_len
        p_compr_avg_len_arr[counter] = p_compressed_avg_len
        p_compr_norm_avg_len_arr[counter] = p_compressed_norm_avg_len
        p_compr_seq_avg_len_arr[counter] = p_compressed_seq_avg_len
        p_avg_no_compression_percentage_arr[counter] = p_no_copression_percentage
        # Correct filter
        c_percentage_arr[counter] = c_percentage
        c_avg_len_arr[counter] = c_avg_answer_len
        c_norm_avg_len_arr[counter] = c_norm_avg_answer_len
        c_compr_avg_len_arr[counter] = c_compressed_avg_len
        c_compr_norm_avg_len_arr[counter] = c_compressed_norm_avg_len
        c_compr_seq_avg_len_arr[counter] = c_compressed_seq_avg_len
        c_avg_no_compression_percentage_arr[counter] = c_no_copression_percentage
        # Correct no prints
        cnp_percentage_arr[counter] = cnp_percentage
        cnp_avg_len_arr[counter] = cnp_avg_answer_len
        cnp_norm_avg_len_arr[counter] = cnp_norm_avg_answer_len
        cnp_compr_avg_len_arr[counter] = cnp_compressed_avg_len
        cnp_compr_norm_avg_len_arr[counter] = cnp_compressed_norm_avg_len
        cnp_compr_seq_avg_len_arr[counter] = cnp_compressed_seq_avg_len
        cnp_avg_no_compression_percentage_arr[counter] = cnp_no_copression_percentage
        # incorrect prints
        ip_percentage_arr[counter] = ip_percentage
        ip_avg_len_arr[counter] = ip_avg_answer_len
        ip_norm_avg_len_arr[counter] = ip_norm_avg_answer_len
        ip_compr_avg_len_arr[counter] = ip_compressed_avg_len
        ip_compr_norm_avg_len_arr[counter] = ip_compressed_norm_avg_len
        ip_compr_seq_avg_len_arr[counter] = ip_compressed_seq_avg_len
        ip_avg_no_compression_percentage_arr[counter] = ip_no_copression_percentage
        # Print and Correct filter
        p_c_percentage_arr[counter] = p_c_percentage
        p_c_avg_len_arr[counter] = p_c_avg_answer_len
        p_c_norm_avg_len_arr[counter] = p_c_norm_avg_answer_len
        p_c_compr_avg_len_arr[counter] = p_c_compressed_avg_len
        p_c_compr_norm_avg_len_arr[counter] = p_c_compressed_norm_avg_len
        p_c_compr_seq_avg_len_arr[counter] = p_c_compressed_seq_avg_len
        p_c_avg_no_compression_percentage_arr[counter] = p_c_no_copression_percentage

        complexity_arr.append(complexity)

    output["complexity"] = complexity_arr
    output["language"] = language_col

    # original data (no filters)
    output["percentages"] = nf_percentage_arr
    output["avg_answer_len"] = nf_avg_len_arr
    output["norm_avg_answer_len"] = nf_norm_avg_len_arr
    output["compressed_avg_len"] = nf_compr_avg_len_arr
    output["compressed_norm_avg_len"] = nf_compr_norm_avg_len_arr
    output["compressed_seq_avg_len"] = nf_compr_seq_avg_len_arr
    output["compressed_seq_avg_len"] = nf_compr_seq_avg_len_arr
    output["no_compression_percentage"] = nf_avg_no_compression_percentage_arr
    # only Print filter applied
    output["p_percentages"] = p_percentage_arr
    output["p_avg_answer_len"] = p_avg_len_arr
    output["p_norm_avg_answer_len"] = p_norm_avg_len_arr
    output["p_compressed_avg_len"] = p_compr_avg_len_arr
    output["p_compressed_norm_avg_len"] = p_compr_norm_avg_len_arr
    output["p_compressed_seq_avg_len"] = p_compr_seq_avg_len_arr
    output["p_no_compression_percentage"] = p_avg_no_compression_percentage_arr
    # only Correct executtion filter applied
    output["c_percentages"] = c_percentage_arr
    output["c_avg_answer_len"] = c_avg_len_arr
    output["c_norm_avg_answer_len"] = c_norm_avg_len_arr
    output["c_compressed_avg_len"] = c_compr_avg_len_arr
    output["c_compressed_norm_avg_len"] = c_compr_norm_avg_len_arr
    output["c_compressed_seq_avg_len"] = c_compr_seq_avg_len_arr
    output["c_no_compression_percentage"] = c_avg_no_compression_percentage_arr
    # Correct no prints
    output["cnp_percentages"] = cnp_percentage_arr
    output["cnp_avg_answer_len"] = cnp_avg_len_arr
    output["cnp_norm_avg_answer_len"] = cnp_norm_avg_len_arr
    output["cnp_compressed_avg_len"] = cnp_compr_avg_len_arr
    output["cnp_compressed_norm_avg_len"] = cnp_compr_norm_avg_len_arr
    output["cnp_compressed_seq_avg_len"] = cnp_compr_seq_avg_len_arr
    output["cnp_no_compression_percentage"] = cnp_avg_no_compression_percentage_arr
    # Incorrect prints
    output["ip_percentages"] = ip_percentage_arr
    output["ip_avg_answer_len"] = ip_avg_len_arr
    output["ip_norm_avg_answer_len"] = ip_norm_avg_len_arr
    output["ip_compressed_avg_len"] = ip_compr_avg_len_arr
    output["ip_compressed_norm_avg_len"] = ip_compr_norm_avg_len_arr
    output["ip_compressed_seq_avg_len"] = ip_compr_seq_avg_len_arr
    output["ip_no_compression_percentage"] = ip_avg_no_compression_percentage_arr
    # Print and Correct execution filters applied
    output["p_c_percentages"] = p_c_percentage_arr
    output["p_c_avg_answer_len"] = p_c_avg_len_arr
    output["p_c_norm_avg_answer_len"] = p_c_norm_avg_len_arr
    output["p_c_compressed_avg_len"] = p_c_compr_avg_len_arr
    output["p_c_compressed_norm_avg_len"] = p_c_compr_norm_avg_len_arr
    output["p_c_compressed_seq_avg_len"] = p_c_compr_seq_avg_len_arr
    output["p_c_no_compression_percentage"] = p_c_avg_no_compression_percentage_arr

    language_filtered_df_dic = {
        "nf": nf_filtered_df,
        "p": p_filtered_df,
        "c": c_filtered_df,
        "cnp": cnp_filtered_df,
        "ip": ip_filtered_df,
        "p_c": p_c_filtered_df,
    }

    return output, language_filtered_df_dic


def analyze_all_languages(rough_all_languages_df, verbose=False):
    language_list = list(rough_all_languages_df["language"].unique())
    analysis_df = pd.DataFrame()

    filtered_df_dict = {}

    for l in language_list:
        l_df = copy.deepcopy(
            rough_all_languages_df[rough_all_languages_df["language"] == l]
        )
        language_out, language_filtered_df_dic = create_language_summary_df(
            rough_language_df=l_df, verbose=verbose, save_language_name=l
        )
        # print("***************************************************************")
        # print("LANGUAGE: ", l)
        # print("***************************************************************")
        # print(language_out)
        analysis_df = pd.concat([analysis_df, language_out])
        filtered_df_dict[l] = language_filtered_df_dic

    return analysis_df, filtered_df_dict


# ----------------------------------------------------------------------------
# The five panels, sliced from 24_PLOTS_paper.ipynb.
# ----------------------------------------------------------------------------

def plot_correct_prints_heatmap(total_df, out_dir):
    """SI Fig 3 top -- correct prints, heatmap by language and complexity.

    Writes ``figure12-top.pdf`` and ``figure12-top.svg`` into ``out_dir``.
    """
    # Filter out complexity 3
    filtered_df = total_df[total_df['complexity'].isin([1, 2, 3])]

    # Pivot the DataFrame to calculate the count of correct executions and the percentage of correct executions with print_code equal to 1
    pivot_table = filtered_df.pivot_table(
        index="complexity",
        columns="language", 
        values="correct_execution",
        aggfunc=["count", lambda x: (x == True).sum() / x.count() * 100],
    )

    # Plotting heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_table["<lambda>"],
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": ""},
        annot_kws={"size": 15}
    )
    plt.title("Percentage of correct prints by Language and Complexity", fontsize=20)
    plt.xlabel("Language",fontsize=20)
    plt.ylabel("Complexity",fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    plt.savefig(str(out_dir / "figure12-top.pdf"), bbox_inches="tight")
    plt.savefig(str(out_dir / "figure12-top.svg"), bbox_inches="tight")
    plt.close('all')


def plot_executions_and_prints(total_df, out_dir):
    """SI Fig 2 top -- correct executions and prints by language and complexity.

    Writes ``figure11-top.pdf`` and ``figure11-top.svg`` into ``out_dir``.
    """
    import numpy as np
    # Filter out complexity 3
    filtered_df = total_df[total_df['complexity'].isin([1, 2, 3])]

    # Group by 'language' and 'complexity' and calculate the percentage of correct executions
    grouped_correct = (
        filtered_df.groupby(["language", "complexity"])["correct_execution"].mean() * 100
    )

    # Count the number of correct executions where print_code is equal to 1
    grouped_print_count = (
        filtered_df[filtered_df["print_code"] == 1]
        .groupby(["language", "complexity"])["correct_execution"]
        .count()
    )

    # Calculate the total number of correct executions per group
    grouped_correct_count = filtered_df.groupby(["language", "complexity"])[
        "correct_execution"
    ].count()

    # Calculate the percentage of print_code for each group
    grouped_print_percentage = (grouped_print_count / grouped_correct_count) * 100

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 12))

    # Width of each bar
    bar_width = 0.40

    # Get the positions for the bars
    x = range(len(grouped_correct))

    # Plotting bars for correct executions
    rects1 = ax.bar(
        x, grouped_correct, width=bar_width, label="Correct Executions", color="lightblue"
    )

    # Plotting bars for percentage of print_code
    rects2 = ax.bar(
        [i + bar_width for i in x],
        grouped_print_percentage,
        width=bar_width,
        label="Print Code = 1",
        color="orange",
    )

    # Adding labels and title
    #ax.set_xlabel("Language and Complexity", fontsize=20)
    ax.set_ylabel("Percentage", fontsize=22)
    ax.set_title(
        "Percentage of Correct Executions and Prints by Language and Complexity",
        fontsize=25  # Increased by 1.5x from the default size (usually 10)
    )
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(
        [
            f"{language}({complexity})"
            for language, complexity in grouped_correct.index
        ],
        rotation=90,
        ha="right",
        fontsize=20  # Increased by 1.5x from the default size (usually 10)
    )
    ax.legend(fontsize=16)

    # Set y-ticks every 5 units from 0 to 100
    plt.yticks(np.arange(0, 105, 5), fontsize=20)

    # Add horizontal grid lines at specific percentages
    for y_value in [2, 5, 10, 25, 50, 100]:
        ax.axhline(y=y_value, color='red', linestyle='-', alpha=0.3, linewidth=7)

    # Add grid
    ax.grid(True, axis='both', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.savefig(str(out_dir / "figure11-top.pdf"), bbox_inches="tight")
    plt.savefig(str(out_dir / "figure11-top.svg"), bbox_inches="tight")
    plt.close('all')


def plot_sequence_overlap(total_df, out_dir):
    """SI Fig 2 bottom -- which sequences each language got right.

    Writes ``figure11_bottom.pdf`` and ``figure11_bottom.svg`` into ``out_dir``.
    """
    def list_of_correct_sequences(language, dataframe):
        df = copy.deepcopy(dataframe[dataframe["language"] == language])
        filtered = df[(df["correct_execution"] == True) & (df["complexity"] != 4)]
        return set(list(filtered["sequence"].unique()))

    mathe_list = list_of_correct_sequences("Mathematica", total_df)
    R_list = list_of_correct_sequences("R", total_df)
    mat_list = list_of_correct_sequences("Matlab", total_df)
    python_list = list_of_correct_sequences("Python", total_df)
    cpp_list = list_of_correct_sequences("C++", total_df)
    arnold_list = list_of_correct_sequences("ArnoldC", total_df)
    java_list = list_of_correct_sequences("JavaScript", total_df)

    from supervenn import supervenn
    sets = [mathe_list, R_list, mat_list, python_list, cpp_list, java_list, arnold_list]

    sets = [mathe_list, R_list, mat_list, python_list, cpp_list, java_list, arnold_list]
    labels = ['Mathematica', 'R', 'Matlab', 'Python', 'Cpp', 'Java', 'ArnoldC']
    plt.figure(figsize=(10, 6))
    supervenn(sets, labels , sets_ordering='minimize gaps')

    plt.savefig(str(out_dir / "figure11_bottom.pdf"), bbox_inches="tight")
    plt.savefig(str(out_dir / "figure11_bottom.svg"), bbox_inches="tight")



def plot_no_compression_by_language(nf_total, out_dir):
    """SI Fig 4 -- no-compression percentage by complexity, with temperature.

    Writes ``figure13.pdf`` and ``figure13.svg`` into ``out_dir``.
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Filter out complexity 3 and rename complexity 4 to 3
    nf_total_filtered = nf_total[nf_total["complexity"] != 4]

    # Create a FacetGrid for multiplot with increased font size
    sns.set(font_scale=1.6) 
    g = sns.FacetGrid(nf_total_filtered, col="language", col_wrap=3, height=4)
    g.set_titles(col_template="{col_name}", fontsize=20)  # Doubled font size from 2 to 4
    g.set_axis_labels("Complexity", "No Compression Percentage")
    for ax in g.axes.flat:
        ax.title.set_fontsize(20)  # Increased font size for title
        ax.xaxis.label.set_size(40)  # X-axis label size
        ax.yaxis.label.set_size(40)  # Y-axis label size
        ax.tick_params(axis='y', labelsize=20)  # Y-axis tick label size

    # Plot the relationship between 'complexity' and 'no_compression_percentage' with shadow indicating 'temperature'
    g.map_dataframe(
        sns.lineplot,
        x="complexity",
        y="no_compression_percentage",
        hue="temperature",
        errorbar="sd",
        marker="o",
        palette="Blues"
    )

    # Set labels and title with increased font size
    g.set_axis_labels("Complexity", "No Compression %", fontsize=20)
    g.set_titles(col_template="{col_name}", fontsize=80)  # Increased font size even more
    g.fig.suptitle(
        "No Compression Percentage by Complexity with Temperature Variation", y=1.05, fontsize=25
    )

    # Adjust legend with increased font size
    g.add_legend(title="Temperature", bbox_to_anchor=(1, 1), loc="upper left", title_fontsize=18, fontsize=18)

    # Set x-axis ticks to only show values 1, 2, and 3
    for ax in g.axes.flat:
        ax.set_xticks([1, 2, 3])
        ax.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig(str(out_dir / "figure13.pdf"), bbox_inches="tight")
    plt.savefig(str(out_dir / "figure13.svg"), bbox_inches="tight")
    plt.close('all')


def plot_no_compression_log(analysis_df, out_dir):
    """SI Fig 3 bottom -- no-compression percentage vs complexity, log scale.

    Writes ``figure12-bottom.pdf`` and ``figure12-bottom.svg`` into ``out_dir``.
    """
    columns_to_plot = [
        # "percentages",
        # "avg_answer_len",
        # "norm_avg_answer_len",
        # "compressed_avg_len",
        # "compressed_norm_avg_len",
        # "compressed_seq_avg_len",
        "no_compression_percentage",
        # "p_percentages",
        # "p_avg_answer_len",
        # "p_norm_avg_answer_len",
        # "p_compressed_avg_len",
        # "p_compressed_norm_avg_len",
        # "p_compressed_seq_avg_len",
        # "p_no_compression_percentage",
        # "c_percentages",
        # "c_avg_answer_len",
        # "c_norm_avg_answer_len",
        # "c_compressed_avg_len",
        # "c_compressed_norm_avg_len",
        # "c_compressed_seq_avg_len",
        # "c_no_compression_percentage",
        # "cnp_percentages",
        # "cnp_avg_answer_len",
        # "cnp_norm_avg_answer_len",
        # "cnp_compressed_avg_len",
        # "cnp_compressed_norm_avg_len",
        # "cnp_compressed_seq_avg_len",
        # "cnp_no_compression_percentage",
        # "ip_percentages",
        # "ip_avg_answer_len",
        # "ip_norm_avg_answer_len",
        # "ip_compressed_avg_len",
        # "ip_compressed_norm_avg_len",
        # "ip_compressed_seq_avg_len",
        # "ip_no_compression_percentage",
        # "p_c_percentages",
        # "p_c_avg_answer_len",
        # "p_c_norm_avg_answer_len",
        # "p_c_compressed_avg_len",
        # "p_c_compressed_norm_avg_len",
        # "p_c_compressed_seq_avg_len",
        # "p_c_no_compression_percentage",
    ]

    # Group the DataFrame by language
    grouped_by_language = analysis_df.groupby("language")

    # Plotting in normal scale
    for column in columns_to_plot:
        plt.figure(figsize=(10, 6))
        plt.title(f"No compression %  vs Complexity (Normal Scale)")
        for language, group in grouped_by_language:
            group = group[group["complexity"].isin([1, 2, 3])]
            plt.plot(group["complexity"], group[column], marker='o', label=language)
        plt.xlabel("Complexity")
        plt.ylabel('Percentage')
        plt.xticks([1, 2, 3])
        plt.legend()
        plt.grid(True)
        plt.close('all')

    # Plotting in log scale
    for column in columns_to_plot:
        plt.figure(figsize=(10, 6))
        plt.title(f"No compression % vs Complexity (Log Scale)")
        for language, group in grouped_by_language:
            group = group[group["complexity"].isin([1, 2, 3])]
            plt.plot(group["complexity"], group[column], marker='o', label=language)
        plt.xlabel("Complexity")
        plt.ylabel('Percentage')
        plt.xticks([1, 2, 3])
        plt.legend()
        plt.grid(True)
        plt.yscale("log")
        plt.savefig(str(out_dir / "figure12-bottom.pdf"), bbox_inches="tight")
        plt.savefig(str(out_dir / "figure12-bottom.svg"), bbox_inches="tight")
        plt.close('all')


def analysis(total_df, write_language_csvs: bool = False):
    """Per-language summary frame plus the unfiltered per-language slices.

    ``write_language_csvs`` restores the published side effect of rewriting the
    seven ``normalized_compressed_*.csv`` in the repository root. Off by default;
    see the module docstring.
    """
    if write_language_csvs:
        return analyze_all_languages(total_df)
    saved = create_language_summary_df.__defaults__
    analysis_df = pd.DataFrame()
    filtered = {}
    for language in list(total_df["language"].unique()):
        one = copy.deepcopy(total_df[total_df["language"] == language])
        out, dic = create_language_summary_df(rough_language_df=one, verbose=False,
                                              save_language_name="")
        analysis_df = pd.concat([analysis_df, out])
        filtered[language] = dic
    del saved
    return analysis_df, filtered


def produce(out_dir: Path | None = None) -> list[Path]:
    """Regenerate all five panels, in the notebook's order.

    The order matters: the first three panels use the swapped complexity tiers,
    then the notebook reloads the data unswapped for the last two. See the module
    docstring.
    """
    out_dir = Path(out_dir) if out_dir is not None else plots_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    swapped = swap_hardest_tiers(concat_all_results())
    plot_correct_prints_heatmap(swapped, out_dir)
    plot_executions_and_prints(swapped, out_dir)
    plot_sequence_overlap(swapped, out_dir)

    fresh = concat_all_results()
    analysis_df, filtered_list_dic = analysis(fresh)
    nf_total = pd.DataFrame()
    for language in filtered_list_dic:
        nf_total = pd.concat([nf_total, filtered_list_dic[language]["nf"]])

    plot_no_compression_by_language(nf_total, out_dir)
    plot_no_compression_log(analysis_df, out_dir)

    written = sorted(p for p in out_dir.iterdir()
                     if p.stem in {"figure11-top", "figure11_bottom", "figure12-top",
                                   "figure12-bottom", "figure13"})
    return written


if __name__ == "__main__":
    for path in produce():
        print(path)

