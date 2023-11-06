
SHELL=/bin/bash

CONDA_ENV_NAME=llm-cooperation
CONDA_ENV_FILE_DEV=./environment.yml
CONDA_ENV_FILE_PRODUCTION=./environment-frozen.yml
CONDA_DIR=$(HOME)/miniforge3
CONDA_BIN=mamba
CONDA_ACTIVATE=source $(CONDA_DIR)/etc/profile.d/conda.sh; source $(CONDA_DIR)/etc/profile.d/mamba.sh; $(CONDA_BIN) activate $(CONDA_ENV_NAME)

book:
	$(CONDA_ACTIVATE); jupyter-book build jupyter-book/

create-results-dir:
	mkdir -p ./results

run: export PYTHONPATH=$(HOME)/.llm-cooperation
run: create-results-dir
	$(CONDA_ACTIVATE); python llm_cooperation/main.py >> results/experiment.log 2>&1

clean:
	$(CONDA_ACTIVATE); jupyter-book clean jupyter-book/

publish: book
	rsync -avz jupyter-book/_build/html/ sphelps.net:/var/www/html/papers/llm-cooperation/

local-module-install:
	$(CONDA_ACTIVATE); pip install -e ./

conda-env-install:
	$(CONDA_BIN) env create -f $(CONDA_ENV_FILE_DEV)

conda-env-production-install:
	$(CONDA_BIN) env create -f $(CONDA_ENV_FILE_PRODUCTION)

env-dev-install: conda-env-install local-module-install

env-production-install: conda-env-production-install local-module-install

conda-update-dev:
	$(CONDA_BIN) env update -f $(CONDA_ENV_FILE_DEV)

conda-update-production:
	$(CONDA_BIN) env export -n $(CONDA_ENV_NAME)| head -n -1 > $(CONDA_ENV_FILE_PRODUCTION)

conda-update: conda-update-dev conda-update-production

install-mamba:
	scripts/install-mamba.sh; echo 'Launch a new shell to continue... '; read

install-pre-commit:
	$(CONDA_ACTIVATE); pre-commit install

install: env-dev-install install-pre-commit jupytext-sync

test-pre-commit:
	$(CONDA_ACTIVATE); pre-commit run --all

test-pytest:
	$(CONDA_ACTIVATE); pytest

test: test-pre-commit test-pytest 

start-blackd:
	$(CONDA_ACTIVATE); blackd

start-notebook:
	$(CONDA_ACTIVATE); jupyter-notebook

jupytext-sync:
	$(CONDA_ACTIVATE); jupytext --sync notebooks/*/*.py
