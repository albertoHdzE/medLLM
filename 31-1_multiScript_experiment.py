"""Script generation: Figures 5 and 6. Forwarder onto :mod:`superarc.scripts`.

This file is kept because it is the path that produced the published figures, and
because every result in the paper traces to this name. The analysis itself now
lives in ``superarc/scripts.py``, where a notebook can import it -- this file
cannot be imported at all, its name beginning with a digit and containing a
hyphen, which is why the repository ended up with a second diverged copy of the
whole analysis inside ``31_multiScript_experiment.ipynb``.

Running it does what it always did::

    python 31-1_multiScript_experiment.py

Output goes to ``superarc.plots_dir()``, which is ``outputs/latest/`` unless
``SUPERARC_PLOTS_DIR`` says otherwise, so a regeneration never writes over the
published artifacts in ``new_plots/``.
"""

from superarc.scripts import main

if __name__ == "__main__":
    main()
