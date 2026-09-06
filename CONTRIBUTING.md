# Contributing

Keep numerical changes traceable to an equation or an explicit experimental protocol. Document normalization, centering, target selection, precision and random-state changes. Avoid substituting PCA for a learned update or modifying a diagnostic to force an expected result.

Run `python -m unittest discover -s tests -v` and the offline smoke experiment before proposing a numerical change. Add a scientific regression test when changing an update identity, conservation property or metric convention. Include the command/configuration and relevant environment metadata in an issue or pull request.

Keep manuscript bibliographic fields as `xxx` until the publication metadata is ready. Do not add identifying paper metadata, credentials, raw private datasets or machine-specific absolute paths. Neural feature recipes must specify weight revisions, preprocessing and output hashes.
