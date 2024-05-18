.PHONY: clean virtualenv lint help
 
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python3
VENV = .venv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

virtualenv: ## Create virtualenv
	rm -rf .venv
	$(PYTHON_INTERPRETER) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(info "Activate with the command 'source .venv/bin/activate'")

add-kernel: ## Add jupyter kernel
	$(PYTHON_INTERPRETER) -m pip install ipykernel
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name=$(NAME)


clean: ## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

pre-commit-install: ## To use the pre-commit hooks
	pre-commit install

lint: ## Lint using ruff
	ruff src/

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'