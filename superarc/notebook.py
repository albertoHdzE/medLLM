"""Display helpers for the verification notebooks.

Presentation only -- nothing here computes a result.

It exists for one reason. The figures are large: the integrated formulae panel
is 24 x 22 inches, and embedding its PNG in a notebook costs about 1.5 MB per
figure. A notebook that shows three of them is a 4 MB file in git, and the
notebooks are committed *with* their outputs on purpose, so that a reviewer can
see that they ran without installing anything.

:func:`preview` resolves that: the full-resolution artifact is written to disk
as always, and the notebook embeds a downscaled copy. What you see is still the
file that was produced, just rendered smaller.
"""

from __future__ import annotations

import io
from pathlib import Path

# Width in pixels of the embedded copy. Wide enough to read the axis labels of a
# six-panel figure, small enough that a notebook stays in the hundreds of
# kilobytes rather than the megabytes.
PREVIEW_WIDTH = 1100


def preview(path: str | Path, width: int = PREVIEW_WIDTH):
    """Embed a downscaled copy of a saved figure.

    ``path`` is the artifact on disk, which is not modified. Returns an
    ``IPython.display.Image`` so a notebook cell can end with it.
    """
    from IPython.display import Image
    from PIL import Image as PILImage

    path = Path(path)
    image = PILImage.open(path)
    if image.width <= width:
        return Image(filename=str(path))

    height = round(image.height * width / image.width)
    resized = image.convert("RGB").resize((width, height), PILImage.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG", optimize=True)
    return Image(data=buffer.getvalue(), format="png")


def figure(name: str, width: int = PREVIEW_WIDTH):
    """Preview a figure by name from the current output directory."""
    from . import plots_dir

    return preview(plots_dir() / name, width=width)
