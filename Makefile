NEXUS_INSIGHTRW_USERNAME?=insight-rw
NEXUS_INSIGHTRW_PWD?=$(shell gopass show nexus/$(NEXUS_INSIGHTRW_USERNAME))
NEXUS_HOST?=nexus.infra.nagra-insight.com
NEXUS_REPO?=pypi-dev
NEXUS_URL=https://$(NEXUS_HOST)/repository/$(NEXUS_REPO)/
PIP_INDEX_URL=https://insight-rw:$(NEXUS_INSIGHTRW_PWD)@$(NEXUS_HOST)/repository/$(NEXUS_REPO)/

# python virtualenv
PYTHON_CMD=python3
VENV_NAME?=.venv
VENV_ACTIVATE=$(VENV_NAME)/bin/activate
PIP_INSTALL?=$(VENV_NAME)/bin/python3 -m pip install --extra-index-url $(PIP_INDEX_URL)

.PHONY: setup
setup:  ## setup development environment
	@test -d $(VENV_NAME) || $(PYTHON_CMD) -m venv $(VENV_NAME) && \
	$(PIP_INSTALL) --upgrade pip && \
	$(PIP_INSTALL) wheel twine && \
	touch $(VENV_ACTIVATE)

.PHONY: release
release: setup  ## release current version to Nexus
	rm -rf dist/*
	. $(VENV_ACTIVATE) && \
	python setup.py bdist_wheel && \
	python -m twine upload \
		--username $(NEXUS_INSIGHTRW_USERNAME) \
		--password $(NEXUS_INSIGHTRW_PWD) \
		--repository-url $(NEXUS_URL) \
		dist/*
