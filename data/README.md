# Data

This repo does not vendor SARDet-100K itself.

Set the dataset root via:

```bash
export SARDET100K_ROOT=/path/to/SARDet_100K
```

or symlink into this directory:

```bash
ln -s /path/to/SARDet_100K data/sardet100k
```

Expected layout:

```
data/sardet100k/
  Annotations/{train,val,test}.json
  JPEGImages/{train,val,test}/
```
