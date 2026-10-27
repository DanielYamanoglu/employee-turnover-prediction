PY := .venv/bin/python

.PHONY: setup train test dashboard
setup:
	uv venv --python 3.12 .venv
	uv pip install -p .venv -r requirements.txt

train:
	PYTHONPATH=src $(PY) -m turnover.train

test:
	$(PY) -m pytest -q

dashboard:
	PYTHONPATH=src .venv/bin/streamlit run app/dashboard.py
