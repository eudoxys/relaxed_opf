#/bin/bash
[ -d .venv ] || python3 venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt
marimo edit notebook.py
