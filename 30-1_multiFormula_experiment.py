"""Formulae generation -- the producer script.

    Percentage of Equivalence between Formulae + Accuracy by Complexity
    Integrated Formulae Analysis: Type Distribution and Volume by Model Complexity

This file is a forwarder. The analysis it used to contain now lives in
:mod:`superarc.formulae`, which is where the notebooks and the CLI reach it.

Why it moved, and why this file still exists
--------------------------------------------
The script could not be imported -- its name begins with a digit and contains a
hyphen -- so anything that wanted the numbers behind these figures had to
recompute them. That is how the repository came to hold two versions of this
analysis: ``30_multiFormula_experiment.ipynb`` carried copies of five of these
functions under the same names, without the corrections this one had received.

The file is kept rather than deleted because it is the path that produced the
published figures, and every result in the paper traces to this name. Deleting
it would break that provenance for the sake of tidiness.

Verified before the move: the module reproduces this script's two figures at
0.0000% pixel difference, and the emitted summary CSV byte for byte. The check
was run with the two plotting calls in the reverse order as well, which exposed
a real latent defect -- the script set matplotlib styles as global statements
*between* the two figures, so calling them the other way round silently
restyled one. Each figure sets its own style now.
"""

from superarc.formulae import (
    accuracy_table,
    add_measures,
    emit_summary_measures,
    equivalence_table,
    load,
    plot_equivalence_and_accuracy,
    plot_integrated,
    print_summary,
    process_formula_data,
)


def main() -> int:
    df = load()
    add_measures(df)

    equiv_df = equivalence_table(df)
    print("\nEquivalence Percentages by Model and Complexity:")
    print(equiv_df.pivot(index="Model", columns="Complexity", values="Equivalence %").round(2))

    print("\nAccuracy by Complexity:")
    print(accuracy_table(df).round(2).T)

    plot_equivalence_and_accuracy(df, equiv_df)
    plot_integrated(df)

    volume_data, total_class_data, accurate_class_data = process_formula_data(df)
    print_summary(volume_data, total_class_data, accurate_class_data)

    emit_summary_measures(df, equiv_df, volume_data, total_class_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
