"""SuperARC-seq: Table 1 and Figures 7-10. Forwarder onto :mod:`superarc.superarc_seq`.

Kept because it is the path that produced the published figures, and because
every result in the paper traces to this name. The analysis lives in
``superarc/superarc_seq.py`` and the metric it measures lives in
``superarc/table1.py``, which owns it.

Running it does what it always did::

    python 34-2_S-ARC_ext.py

Output goes to ``superarc.plots_dir()``, which is ``outputs/latest/`` unless
``SUPERARC_PLOTS_DIR`` says otherwise, so a regeneration never writes over the
published artifacts in ``new_plots/``.
"""

from superarc.superarc_seq import main

if __name__ == "__main__":
    main()
