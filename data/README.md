# Data Layout

Use immutable, append-only data practices.

- `raw/`: vendor-native files. Never edit manually.
- `bronze/`: normalized vendor data with minimal cleaning.
- `silver/`: research-ready bars, events, and reference data.
- `features/`: point-in-time feature matrices.

Commit schemas, manifests, and small fixtures. Track large datasets with DVC or
external object storage.
