#!/bin/bash
python encode.py --users --items  # To get the encodings (npz)
python lr.py data/dummy/X-ui.npz  # To get results (txt)
python fm.py data/dummy/X-ui.npz
