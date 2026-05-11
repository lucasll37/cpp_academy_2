.PHONY: deps setup install clean test run help

.DEFAULT_GOAL := help

# Python environment configuration
PYTHON_DIR := $(shell pwd)
PYTHON_VENV := $(PYTHON_DIR)/.venv
PYTHON_BIN := $(PYTHON_VENV)/bin/python
PIP_BIN := $(PYTHON_VENV)/bin/pip
PYTHON_VERSION := python3.12

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================
# Python Environment Targets
# ============================================

deps: ## Install Python 3.12 if not available
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Checking Python 3.12 Installation...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@if ! command -v $(PYTHON_VERSION) &> /dev/null && \
	    ! test -x /usr/bin/python3.12 && \
	    ! test -x /usr/local/bin/python3.12; then \
		echo "$(YELLOW)Python 3.12 not found. Installing...$(NC)"; \
		sudo apt update && \
		sudo apt install -y software-properties-common && \
		sudo add-apt-repository -y ppa:deadsnakes/ppa && \
		sudo apt update && \
		sudo apt install -y python3.12 python3.12-venv python3.12-dev && \
		echo "$(GREEN)✓ Python 3.12 installed successfully$(NC)"; \
	else \
		echo "$(GREEN)✓ Python 3.12 already installed$(NC)"; \
	fi
	@test -x /usr/bin/python3.12     && /usr/bin/python3.12 --version     || \
	 test -x /usr/local/bin/python3.12 && /usr/local/bin/python3.12 --version || \
	 $(PYTHON_VERSION) --version

setup: deps ## Create Python virtual environment
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Creating Python Environment...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@if [ ! -d "$(PYTHON_VENV)" ]; then \
		echo "$(YELLOW)Creating virtual environment with $(PYTHON_VERSION)...$(NC)"; \
		$(PYTHON_VERSION) -m venv $(PYTHON_VENV); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	else \
		echo "$(GREEN)✓ Virtual environment already exists$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)To activate manually:$(NC)"
	@echo "  source $(PYTHON_VENV)/bin/activate"

install: setup ## Install Python dependencies
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Installing Python Dependencies...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(YELLOW)Upgrading pip...$(NC)"
	@$(PIP_BIN) install --upgrade pip
	@echo ""
	@echo "$(YELLOW)Installing requirements...$(NC)"
	@$(PIP_BIN) install -r $(PYTHON_DIR)/requirements.txt
	@echo ""
	@echo "$(GREEN)✓ Python dependencies installed$(NC)"

run: ## Run pilotBT.py with the dummy input
	@echo "$(BLUE)Running pilotBT.py...$(NC)"
	@$(PYTHON_BIN) -c "\
import sys; sys.path.insert(0, '.'); \
from pilotBT import PilotBT; \
import json; \
m = PilotBT(); m.load(); \
dummy = PilotBT.make_dummy_input(); \
print(); print('=== predict() output ===\n'); \
out = m.predict(dummy); \
[print(f'  {k}: {v.flatten()[0]:.4f}') for k, v in out.items()]"

test: ## Validate pilotBT.py against the server schema
	@echo "$(BLUE)Validating pilotBT.py...$(NC)"
	@$(PYTHON_BIN) test_model.py pilotBT.py --predict

clean: ## Remove Python virtual environment
	@echo "$(YELLOW)Removing Python virtual environment...$(NC)"
	@rm -rf $(PYTHON_VENV)
	@rm -rf $(PYTHON_DIR)/__pycache__
	@rm -rf $(PYTHON_DIR)/*.pyc
	@find $(PYTHON_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Python environment cleaned$(NC)"

# ============================================
# Misc Targets
# ============================================

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'