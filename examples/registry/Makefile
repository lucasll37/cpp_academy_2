.PHONY: help
.PHONY: python-env python-run python-demo python-clean
.PHONY: minio-up minio-down minio-logs minio-reset minio-status
.PHONY: user-create user-list user-keygen

.DEFAULT_GOAL := help

# ============================================
# Variables
# ============================================

PWD := $(shell pwd)

# AIStor / Docker configuration
COMPOSE_FILE      := docker-compose.yml
MINIO_CONTAINER   := aistor-server
MC_IMAGE          := quay.io/minio/aistor/mc:latest
MINIO_ENDPOINT    := localhost:9000
MINIO_CONSOLE     := http://localhost:9001
DATA_DIR          := ./minio-data

# Lê credenciais do .env
MINIO_ROOT_USER     := $(shell grep ^MINIO_ROOT_USER .env 2>/dev/null | cut -d= -f2)
MINIO_ROOT_PASSWORD := $(shell grep ^MINIO_ROOT_PASSWORD .env 2>/dev/null | cut -d= -f2)

# Macro: roda comandos mc em container temporário autenticado
define run-mc
	docker run --rm --network host \
		--entrypoint /bin/sh \
		-v $(PWD)/init:/init \
		$(MC_IMAGE) -c " \
			mc alias set local http://$(MINIO_ENDPOINT) $(MINIO_ROOT_USER) $(MINIO_ROOT_PASSWORD) \
			&& $(1)"
endef

# Colors for output
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[1;33m
BLUE   := \033[0;34m
NC     := \033[0m


# ============================================
# Help
# ============================================

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)AIStor Free — Local Environment$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "$(YELLOW)AIStor:$(NC)"
	@grep -E '^minio-[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Python:$(NC)"
	@grep -E '^python-[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Users:$(NC)"
	@grep -E '^user-[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(NC) %s\n", $$1, $$2}'
	@echo ""


# ============================================
# AIStor Targets
# ============================================

minio-up: ## Start AIStor
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Starting AIStor...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@[ -f .env ] || { cp .env.example .env; \
		echo "$(YELLOW)⚠  .env criado a partir de .env.example$(NC)"; \
		echo "$(RED)   Edite .env com suas credenciais e rode novamente.$(NC)"; exit 1; }
	@[ -f minio.license ] || { \
		echo "$(RED)✗ minio.license não encontrado!$(NC)"; \
		echo "  Obtenha a licença gratuita em: https://min.io/pricing"; exit 1; }
	@docker compose -f $(COMPOSE_FILE) up -d
	@echo ""
	@echo "$(GREEN)✓ AIStor is up$(NC)"
	@echo "  API:     http://$(MINIO_ENDPOINT)"
	@echo "  Console: $(MINIO_CONSOLE)"

minio-down: ## Stop AIStor containers
	@echo "$(YELLOW)Stopping AIStor...$(NC)"
	@docker compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)✓ AIStor stopped$(NC)"

minio-logs: ## Follow AIStor container logs
	@docker compose -f $(COMPOSE_FILE) logs -f $(MINIO_CONTAINER)

minio-status: ## Show container status and API health
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)AIStor Status$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@docker compose -f $(COMPOSE_FILE) ps
	@echo ""
	@echo "$(YELLOW)Health check:$(NC)"
	@curl -sf http://$(MINIO_ENDPOINT)/minio/health/live \
		&& echo "$(GREEN)✓ AIStor API is healthy$(NC)" \
		|| echo "$(RED)✗ AIStor API is not responding$(NC)"

minio-reset: minio-down ## ⚠ Stop AIStor and wipe ALL data (irreversible)
	@echo "$(RED)========================================$(NC)"
	@echo "$(RED)Resetting AIStor — ALL DATA WILL BE LOST$(NC)"
	@echo "$(RED)========================================$(NC)"
	@docker compose -f $(COMPOSE_FILE) down -v
	@sudo rm -rf $(DATA_DIR)
	@echo "$(GREEN)✓ AIStor data wiped$(NC)"


minio-apply-policies: ## Reapply all IAM policies (use after editing policy JSON files)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Applying policies...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@$(call run-mc, \
		mc admin policy create local policy-shared   /init/policy-shared.json   && \
		mc admin policy create local policy-readonly /init/policy-readonly.json && \
		mc admin policy create local policy-personal /init/policy-personal.json)
	@echo "$(GREEN)✓ Policies aplicadas$(NC)"

# ============================================
# User Management Targets
# ============================================

user-create: ## Create user with policies and personal folder. Usage: make user-create USER=lucas PASS=senha123
	@[ -n "$(USER)" ] && [ -n "$(PASS)" ] || { \
		echo "$(RED)Erro: informe USER e PASS$(NC)"; \
		echo "  Uso: make user-create USER=lucas PASS=senha123"; exit 1; }
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Criando usuário '$(USER)'...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@$(call run-mc, \
		mc admin policy create local policy-shared   /init/policy-shared.json   || true && \
		mc admin policy create local policy-readonly /init/policy-readonly.json || true && \
		mc admin policy create local policy-personal /init/policy-personal.json || true && \
		mc admin user add local $(USER) $(PASS) && \
		mc admin policy attach local policy-shared   --user $(USER) && \
		mc admin policy attach local policy-readonly --user $(USER) && \
		mc admin policy attach local policy-personal --user $(USER))
	@docker run --rm --network host \
		--entrypoint /bin/sh \
		$(MC_IMAGE) -c " \
			mc alias set local http://$(MINIO_ENDPOINT) $(MINIO_ROOT_USER) $(MINIO_ROOT_PASSWORD) && \
			printf 'placeholder' | mc pipe local/users/$(USER)/.keep"
	@echo "$(GREEN)✓ Usuário '$(USER)' criado$(NC)"
	@echo "  Acesso a: shared/  readonly/  users/$(USER)/"

user-list: ## List all registered users
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Usuários cadastrados$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@$(call run-mc, mc admin user list local)


user-keygen: ## Generate service account for a user. Usage: make user-keygen USER=lucas
	@[ -n "$(USER)" ] || { \
		echo "$(RED)Erro: informe USER$(NC)"; \
		echo "  Uso: make user-keygen USER=lucas"; exit 1; }
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Gerando access key para '$(USER)'...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@$(call run-mc, mc admin user svcacct add local $(USER))
	@echo ""
	@echo "$(YELLOW)⚠  Salve o Secret Key agora — ele não será exibido novamente.$(NC)"
	@echo "  Adicione no .env:"
	@echo "    MINIO_SVC_ACCESS_KEY=<access-key>"
	@echo "    MINIO_SVC_SECRET_KEY=<secret-key>"
	@echo "    MINIO_SVC_USERNAME=$(USER)"

# ============================================
# Python Targets
# ============================================

PYTHON_DIR     := python
PYTHON_VENV    := $(PYTHON_DIR)/.venv
PIP            := $(PYTHON_VENV)/bin/pip
PYTHON         := $(PYTHON_VENV)/bin/python
PYTHON_VERSION := python3.12


python-env: ## Create .venv and install dependencies
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Setting up Python environment...$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@[ -d $(PYTHON_VENV) ] || $(PYTHON_VERSION) -m venv $(PYTHON_VENV)
	@$(PIP) install --upgrade pip --quiet
	@$(PIP) install -r $(PYTHON_DIR)/requirements.txt --quiet
	@echo "$(GREEN)✓ Python environment ready$(NC)"
	@echo "  Activate: source $(PYTHON_VENV)/bin/activate"

python-run: python-env ## Run bucket_demo.py
	@echo "$(BLUE)Running bucket_demo.py...$(NC)"
	@$(PYTHON) $(PYTHON_DIR)/bucket_demo.py

python-demo: python-env ## Upload 10 files to personal folder and list them
	@echo "$(BLUE)Running personal_folder_demo.py...$(NC)"
	@$(PYTHON) $(PYTHON_DIR)/personal_folder_demo.py

python-clean: ## Remove Python virtual environment
	@rm -rf $(PYTHON_VENV)
	@find $(PYTHON_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Python environment removed$(NC)"