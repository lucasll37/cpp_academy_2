.PHONY: clean configure build install package help coverage coverage-open
.PHONY: run-server run-client test test-unit test-integration test-verbose
.PHONY: docs docs-open docs-clean
.PHONY: python-env python-install create-models python-clean
.PHONY: docker-build-cpu docker-build-gpu docker-run-cpu docker-run-gpu

.DEFAULT_GOAL := help

# Custom variables
PWD := $(shell pwd)
BUILD_DIR := ./build
DIST_DIR := $(PWD)/../dist

# Determine number of parallel jobs for Ninja (half of available cores)
NINJA_JOBS := $(shell expr $$(nproc) / 2)

# Build configuration
BUILD_TYPE := Debug

# Documentation configuration
DOCS_DIR := docs/doxygen/generated
DOXYFILE := docs/doxygen/Doxyfile

# Python environment configuration
PYTHON_DIR := models
PYTHON_VENV := $(PYTHON_DIR)/.venv
PYTHON_BIN := $(PYTHON_VENV)/bin/python
PIP_BIN := $(PYTHON_VENV)/bin/pip
PYTHON_VERSION := python3.12

# Docker configuration
DOCKER_WORKER_IMAGE := ml-inference-server
DOCKER_TAG := latest

# Browser configuration for opening documentation
BROWSER := explorer.exe

# Coverage configuration
COVERAGE_DIR := $(BUILD_DIR)/coverage-report

# Clang configuration
FORMAT_SOURCES := $(shell find core tests -name '*.cpp' -o -name '*.hpp')
TIDY_SOURCES := 'core/.*\.cpp$$'


# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color


# ============================================
# C++ Build Targets
# ============================================

clean: ## Clean all generated build files in the project.
	rm -rf $(BUILD_DIR)/
	rm -rf ./subprojects/packagecache

configure: ## Configure the project for building.
	mkdir -p $(BUILD_DIR)/
	conan install ./ \
		--build=missing \
		--settings=build_type=$(BUILD_TYPE) \
		-c tools.system.package_manager:mode=install \
		-c tools.system.package_manager:sudo=True

	meson setup --reconfigure \
		--backend ninja \
		--buildtype debug \
		--buildtype $(shell echo $(BUILD_TYPE) | tr '[:upper:]' '[:lower:]') \
		--native-file $(BUILD_DIR)/conan_meson_native.ini \
		--prefix=$(DIST_DIR) \
		--libdir=$(DIST_DIR)/lib \
		-Db_coverage=true \
		-Dpkg_config_path=$(DIST_DIR)/lib/pkgconfig:$(BUILD_DIR) \
		$(BUILD_DIR)/ .

build: ## Build all targets in the project.
	meson compile -C $(BUILD_DIR) -j$(NINJA_JOBS)

install: ## Install all targets in the project.
	meson install -C $(BUILD_DIR)
	mkdir -p $(DIST_DIR)/models
	cp -r ./models/. $(DIST_DIR)/models/

package: ## Package the project using conan.
	conan create ./ \
		--build=missing \
		--settings=compiler.cppstd=17 \
		--settings=build_type=Debug

	conan create ./ \
		--build=missing \
		--settings=compiler.cppstd=17 \
		--settings=build_type=Release


# ============================================
# Execution Targets
# ============================================

run-server: ## Run the server.
	$(BUILD_DIR)/core/server/AsaMiia --address 0.0.0.0:50052 --models-dir ./models --threads 8

run-client: ## Run the client example. --address localhost:50052
	$(BUILD_DIR)/core/client/AsaMiiaClient --models-dir ./models --address localhost:50052


# ============================================
# Tests Targets
# ============================================

test: build ## Run all tests.
	meson test -C $(BUILD_DIR) --test-args '--gtest_output=json:test_results.json --gtest_print_time=1 --gtest_color=yes'

test-unit: build ## Run only unit tests (sem worker).
	meson test -C $(BUILD_DIR) --suite unit --test-args '--gtest_output=json:test_results.json --gtest_print_time=1 --gtest_color=yes'

test-integration: build ## Run integration tests (requer worker rodando).
	meson test -C $(BUILD_DIR) --suite integration --test-args '--gtest_output=json:test_results.json --gtest_print_time=1 --gtest_color=yes'

test-verbose: build ## Run tests com output detalhado.
	meson test -C $(BUILD_DIR) --verbose --print-errorlogs --test-args '--gtest_output=json:test_results.json --gtest_print_time=1 --gtest_color=yes'


# ============================================
# Coverage Targets
# ============================================

coverage: test ## Gera relatório de cobertura de testes
	geninfo ./build \
		--output-filename ./build/coverage.info \
		--ignore-errors mismatch,mismatch \
		--ignore-errors gcov,gcov \
		--memory 0

	lcov --remove ./build/coverage.info \
		'/usr/*' \
		'*/build/*' \
		'*/test*' \
		"${HOME}/.conan2/*" \
		'*/.conan2/*' \
		'*/google/protobuf/*' \
		'*/protobuf/*' \
		'*/models/*' \
		'*/python/*' \
		--output-file ./build/coverage.info \
		--ignore-errors unused

	genhtml ./build/coverage.info \
		--output-directory ./build/coverage-report

	@echo "Relatório em: build/coverage-report/index.html"

coverage-open: ## Abre relatório de cobertura no browser
	-@$(BROWSER) $$(wslpath -w $(COVERAGE_DIR)/index.html) > /dev/null 2>&1 || true


# ============================================
# Documentation Targets
# ============================================

docs: ## Gera documentação com Doxygen
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Generating Doxygen Documentation...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@command -v doxygen > /dev/null 2>&1 || { echo "$(RED)doxygen not found. Run: sudo apt install doxygen graphviz$(NC)"; exit 1; }
	@doxygen $(DOXYFILE)
	@echo "$(GREEN)✓ Documentation generated: $(DOCS_DIR)/html/index.html$(NC)"

docs-open: ## Abre documentação no browser
	-@$(BROWSER) $$(wslpath -w $(DOCS_DIR)/html/index.html) > /dev/null 2>&1 || true

docs-clean: ## Remove documentação gerada
	@rm -rf $(DOCS_DIR)
	@echo "$(GREEN)✓ Documentation cleaned$(NC)"


# ============================================
# Python Environment Targets
# ============================================

python-setup: ## Install Python 3.12 if not available
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

python-env: python-setup ## Create Python virtual environment
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Creating Python Environment...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@mkdir -p models
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
	@$(PIP_BIN) install -r python/requirements.txt
	@echo ""
	@echo "$(GREEN)✓ Python dependencies installed$(NC)"
	@echo ""
	@echo "Installed packages:"
	@$(PIP_BIN) list | grep -E "onnx|numpy|protobuf"

create-models: python-install ## Create test ONNX models
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Creating Test ONNX Models...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@mkdir -p models
	@$(PYTHON_BIN) python/scripts/create_test_models.py
	@echo ""
	@if [ -d "python/models" ] && [ "$$(ls -A python/models 2>/dev/null)" ]; then \
		echo "$(YELLOW)Copying Python models...$(NC)"; \
		cp -v python/models/* models/ 2>/dev/null; \
		echo "$(GREEN)✓ Python models copied$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)✓ Models created!$(NC)"
	@echo ""
	@echo "Created models:"
	@ls -lh models/ 2>/dev/null || echo "  No models found"

python-clean: ## Remove Python virtual environment
	@echo "$(YELLOW)Removing Python virtual environment...$(NC)"
	@rm -rf $(PYTHON_VENV)
	@rm -rf $(PYTHON_DIR)/__pycache__
	@rm -rf $(PYTHON_DIR)/*.pyc
	@find $(PYTHON_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Python environment cleaned$(NC)"


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
	docker run -d --rm -p 50052:50052 --gpus all -v $(PWD)/models:/app/models:ro $(DOCKER_WORKER_IMAGE):$(DOCKER_TAG)-gpu


# ============================================
# clang-tidy targets
# ============================================

tidy: ## Analisa código com clang-tidy
	run-clang-tidy -p $(BUILD_DIR) $(TIDY_SOURCES)

# tidy-fix: ## Aplica fixes automáticos do clang-tidy
# 	run-clang-tidy --use-color -p $(BUILD_DIR) -fix $(TIDY_SOURCES)

tidy-file: ## Roda clang-tidy em FILE= específico.
	clang-tidy --use-color -p $(BUILD_DIR) $(FILE)


# ============================================
# clang-format targets
# ============================================

format: ## Formata todo o código fonte in-place
	clang-format -i $(FORMAT_SOURCES)

format-check: ## Verifica formatação sem modificar (para CI)
	clang-format --dry-run --Werror $(FORMAT_SOURCES)

format-file: ## Formata FILE= específico. Ex: make format-file FILE=core/inference/src/onnx_backend.cpp
	@test -n "$(FILE)" || (echo "uso: make format-file FILE=<arquivo>" && exit 1)
	clang-format -i $(FILE)

format-diff: ## Mostra o diff do que seria formatado sem aplicar
	@test -n "$(FILE)" || (echo "uso: make format-diff FILE=<arquivo>" && exit 1)
	clang-format $(FILE) | diff $(FILE) - || true


# ============================================
# Misc Targets
# ============================================

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'