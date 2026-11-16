# Local source data

Place `train.csv`, `test.csv`, and `sample_submission.csv` in `data/raw/`.
The delivered local folder contains copies of the user's supplied files. They are
excluded from Git and the source-only archive. Dataset provenance, license, and
permission to redistribute were not supplied; no public redistribution is assumed.
Files are fingerprinted by SHA-256 in `reports/run_manifest.json`.

The target is the supplied binary `cancelled` label. It does not establish which
party initiated cancellation. Timestamp timezone and distance units are unknown.
No external data was added.
