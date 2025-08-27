# -------- Makefile (safe quoting + cross‑platform) --------
SHELL := /bin/bash
PY ?= .venv/bin/python
# Quote PYTHONPATH so spaces in the absolute path don't break commands
export PYTHONPATH := "$(shell pwd)"

OS := $(shell uname -s 2>/dev/null || echo Windows)
BREW := $(shell command -v brew 2>/dev/null)
APT := $(shell command -v apt-get 2>/dev/null)

.PHONY: help install freeze train serve drift verify-xgb check-env docker-build docker-run lint test clean

help:
	@echo "Targets:"
	@echo "  make install      - install deps (+ OpenMP on macOS if needed)"
	@echo "  make freeze       - write exact versions to requirements.txt"
	@echo "  make lint         - lint the code"
	@echo "  make test         - run tests"
	@echo "  make train        - train model (uses configs/training.yaml)"
	@echo "  make serve        - start FastAPI on :8000"
	@echo "  make drift        - run drift check (if script present)"
	@echo "  make clean        - remove temporary files"
	@echo "  make verify-xgb   - import-test xgboost & show version"
	@echo "  make check-env    - show Python & path info"
	@echo "  make docker-build - build Docker image"
	@echo "  make docker-run   - run Docker image"

install:
	$(PY) -m pip install -U pip wheel setuptools
	$(PY) -m pip install -r requirements.txt || true

install-dev:
	$(PY) -m pip install -r requirements-dev.txt || true
	$(PY) -m pip install pre-commit
	.venv/bin/pre-commit install
	# --- OpenMP/XGBoost runtime (macOS) ---
	@if [ "$(OS)" = "Darwin" ]; then \
	  if [ -n "$(BREW)" ]; then \
	    echo "🔧 Ensuring OpenMP runtime via Homebrew..."; \
	    brew list llvm-openmp >/dev/null 2>&1 || brew list libomp >/dev/null 2>&1 || brew install llvm-openmp || brew install libomp; \
	  else \
	    echo "ℹ️  Homebrew not found; if xgboost fails to import, install brew+llvm-openmp/libomp."; \
	  fi; \
	fi
	# --- Linux note (usually already available via libgomp) ---
	@if [ "$(OS)" = "Linux" ] && [ -n "$(APT)" ]; then \
	  echo "ℹ️  On Debian/Ubuntu, libgomp1 provides OpenMP: sudo apt-get install -y libgomp1"; \
	fi

freeze:
	$(PY) -m pip freeze > requirements.txt

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest

train:
	$(PY) -m scripts.train --config configs/training.yaml

serve:
	PYTHONPATH=$(PYTHONPATH) .venv/bin/uvicorn api.app:app --reload --port 8000

drift:
	@if [ -f scripts/detect_drift.py ]; then \
	  PYTHONPATH=$(PYTHONPATH) $(PY) -m scripts.detect_drift --config configs/drift.yaml; \
	else \
	  echo "No scripts/detect_drift.py (skipping)"; \
	fi

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete

verify-xgb:
	@$(PY) -c "import xgboost; print('✅ xgboost:', xgboost.__version__)"

check-env:
	@echo "OS: $(OS)"
	@echo "CWD: $(pwd)"
	@echo "PYTHON: $(which $(PY))"
	@echo "PYTHONPATH: $(PYTHONPATH)"
	@$(PY) -V
	@$(PY) - <<'PY'
		import sys, os, importlib.util
		print("Top sys.path:", sys.path[:3])
		print("Can import 'src'? ", importlib.util.find_spec('src') is not None)
	PY


docker-build:
	docker build -t churn-api:latest .

docker-run:
	docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" churn-api:latest
