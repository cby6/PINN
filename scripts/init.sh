#!/bin/bash

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
rm -rf .venv
uv venv .venv --python 3.11
. .venv/bin/activate
uv pip install --upgrade --quiet pip wheel setuptools
uv pip install --editable .