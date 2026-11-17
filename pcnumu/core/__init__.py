"""Shared library for evaluating pc-numu TransformerEE models.

  data      load and align test predictions; model folders; train.log; provenance hashes
  metrics   metric definitions: bias, resolution, windows, ellipse, error maps, peak pull, intervals
  regional  regional comparison: windows, migration counts, paired bootstrap, count checks
  style     the house plot style used by every figure
  plots     plots shared by several commands
  report    JSON and Markdown writers

Commands that use it are in pcnumu/eval/; see pcnumu/README.md.
"""
