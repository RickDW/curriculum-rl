#!/bin/bash
conda activate base
xvfb-run -a /opt/conda/bin/python "$@"