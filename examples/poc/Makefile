.PHONY: clean configure build install package help
.PHONY: run-worker run-client test
.PHONY: python-env python-install create-models

.DEFAULT_GOAL := help

# Custom variables
PWD := $(shell pwd)
BUILD_DIR := ./build
DIST_DIR := $(PWD)/../dist

# Build configuration
BUILD_TYPE := Debug
CONAN_BUILD_TYPE := Debug

# Python environment configuration
PYTHON_DIR := python
PYTHON_VENV := $(PYTHON_DIR)/.venv
PYTHON_BIN := $(PYTHON_VENV)/bin/python
PIP_BIN := $(PYTHON_VENV)/bin/pip
PYTHON_VERSION := python3.12

# Docker configuration
DOCKER_WORKER_IMAGE := ml-inference-worker
DOCKER_TAG := latest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================
# C++ Build Targets
# ============================================

configure: ## Configure the project for building.
	mkdir -p $(BUILD_DIR)/
	conan install ./ \
		--build=missing \
		--settings=build_type=$(CONAN_BUILD_TYPE) \
		--remote=conancenter
	meson setup --reconfigure \
		--backend ninja \
		--buildtype debug \
		--prefix=$(DIST_DIR) \
		--libdir=$(DIST_DIR)/lib \
		-Dpkg_config_path=$(DIST_DIR)/lib/pkgconfig:$(BUILD_DIR) \
		$(BUILD_DIR)/ .

build: ## Build all targets in the project.
	meson compile -C $(BUILD_DIR)

install: ## Install all targets in the project.
	meson install -C $(BUILD_DIR)
	mkdir -p $(DIST_DIR)/models
	cp -r ./models/* $(DIST_DIR)/models/ 2>/dev/null || true

package: ## Package the project using conan.
	VENV_PYTHON=$(PWD)/python/.venv/bin/python conan create ./ \
		--build=missing \
		--settings=compiler.cppstd=17 \
		--settings=build_type=Debug
	VENV_PYTHON=$(PWD)/python/.venv/bin/python conan create ./ \
		--build=missing \
		--settings=compiler.cppstd=17 \
		--settings=build_type=Release

clean: ## Clean all generated build files in the project.
	rm -rf $(BUILD_DIR)/
	rm -rf ./subprojects/packagecache

# Additional targets for development
run-worker: ## Run the worker server.
	$(BUILD_DIR)/cpp/worker/AsaMiia --address 0.0.0.0:50052 --threads 8

run-client: ## Run the client example.
	$(BUILD_DIR)/cpp/client/AsaMiiaClient localhost:50052

test: build ## Run all tests.
	meson test -C $(BUILD_DIR)

test-unit: build ## Run only unit tests (sem worker).
	meson test -C $(BUILD_DIR) --suite unit

test-integration: build ## Run integration tests (requer worker rodando).
	meson test -C $(BUILD_DIR) --suite integration

test-verbose: build ## Run tests com output detalhado.
	meson test -C $(BUILD_DIR) --verbose

test-ci: build ## Run tests para CI (unit apenas).
	meson test -C $(BUILD_DIR) --suite unit --print-errorlogs

# ============================================
# Python Environment Targets
# ============================================

python-setup: ## Install Python 3.12 if not available
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Checking Python 3.12 Installation...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@if ! command -v $(PYTHON_VERSION) &> /dev/null; then \
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
	@$(PYTHON_VERSION) --version

python-env: python-setup ## Create Python virtual environment
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

python-install: python-env ## Install Python dependencies
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
	@echo ""
	@echo "Installed packages:"
	@$(PIP_BIN) list | grep -E "onnx|numpy|protobuf"

python-clean: ## Remove Python virtual environment
	@echo "$(YELLOW)Removing Python virtual environment...$(NC)"
	@rm -rf $(PYTHON_VENV)
	@rm -rf $(PYTHON_DIR)/__pycache__
	@rm -rf $(PYTHON_DIR)/*.pyc
	@find $(PYTHON_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Python environment cleaned$(NC)"

python-test: python-install ## Test Python environment
	@echo "$(BLUE)Testing Python environment...$(NC)"
	@$(PYTHON_BIN) --version
	@echo ""
	@echo "Testing imports:"
	@$(PYTHON_BIN) -c "import onnx; print('✓ onnx:', onnx.__version__)"
	@$(PYTHON_BIN) -c "import numpy; print('✓ numpy:', numpy.__version__)"
	@$(PYTHON_BIN) -c "import onnxruntime; print('✓ onnxruntime:', onnxruntime.__version__)"
	@echo ""
	@echo "$(GREEN)✓ Python environment is working!$(NC)"

create-models: python-install ## Create test ONNX models
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Creating Test ONNX Models...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@mkdir -p models
	@$(PYTHON_BIN) $(PYTHON_DIR)/scripts/create_test_models.py
	@echo ""
	@if [ -d "$(PYTHON_DIR)/models" ] && [ "$$(ls -A $(PYTHON_DIR)/models 2>/dev/null)" ]; then \
		echo "$(YELLOW)Copying Python models...$(NC)"; \
		cp -v $(PYTHON_DIR)/models/* models/ 2>/dev/null; \
		echo "$(GREEN)✓ Python models copied$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)✓ Models created!$(NC)"
	@echo ""
	@echo "Created models:"
	@ls -lh models/ 2>/dev/null || echo "  No models found"

# ============================================
# Docker Build Targets
# ============================================

docker-build-cpu: ## Build worker Docker image (CPU only)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Building Worker Docker Image (CPU)...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@chmod +x scripts/worker_docker.sh
	@./scripts/worker_docker.sh --cpu --tag $(DOCKER_TAG)
	@echo ""
	@echo "$(GREEN)✓ Worker CPU image built!$(NC)"
	@echo ""
	@echo "Image: $(DOCKER_WORKER_IMAGE):$(DOCKER_TAG)-cpu"

docker-build-gpu: ## Build worker Docker image (GPU support)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Building Worker Docker Image (GPU)...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@chmod +x scripts/worker_docker.sh
	@./scripts/worker_docker.sh --gpu --tag $(DOCKER_TAG)
	@echo ""
	@echo "$(GREEN)✓ Worker GPU image built!$(NC)"
	@echo ""
	@echo "Image: $(DOCKER_WORKER_IMAGE):$(DOCKER_TAG)-gpu"

docker-run-cpu: ## Run worker Docker container (CPU)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Running Worker Docker Container (CPU)...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	docker run -d --rm -p 50052:50052 -v $(PWD)/models:/app/models:ro $(DOCKER_WORKER_IMAGE):$(DOCKER_TAG)-cpu

docker-run-gpu: ## Run worker Docker container (GPU)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Running Worker Docker Container (GPU)...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	docker run -d --rm -p 50052:50052 --gpus all -v $(PWD)/models:/models:ro $(DOCKER_WORKER_IMAGE):$(DOCKER_TAG)-gpu


help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'