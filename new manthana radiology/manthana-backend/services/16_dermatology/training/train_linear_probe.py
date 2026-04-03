"""
Optional V2: frozen EfficientNet-B4 embeddings + sklearn linear probe on SCIN-mapped labels.

Not a full fine-tune — CPU-friendly. Populate image_paths/labels from SCIN metadata per
https://github.com/google-research-datasets/scin and your legal review of the SCIN license.
"""

from __future__ import annotations

# Placeholder — implement when SCIN pipeline is wired.
def main() -> None:
    raise SystemExit(
        "train_linear_probe: add SCIN CSV parsing and label mapping to DERM_CLASSES, then run."
    )


if __name__ == "__main__":
    main()
