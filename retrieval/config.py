"""Tunables, in one place, with where each number came from.

MIN_COSINE is calibration dependent. It is the value measured for the bundled
HashingEmbedder on the bundled corpus by `eval/calibrate.py`, which found the
largest unsafe cosine at 0.306 and the smallest safe one at 0.314.

The production system this is extracted from gates at 0.28, but it embeds with
Amazon Titan at 1,024 dimensions where the same texts sit further apart. Carrying
a threshold across embedders is how a system ends up refusing good questions and
answering bad ones, so recalibrate whenever the embedder or the corpus changes:

    python3 eval/calibrate.py
"""
MIN_COSINE = 0.31
