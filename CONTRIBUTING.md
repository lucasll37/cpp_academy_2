# **CONTRIBUTING**

# Guia de Contribuição – AI Model Runtime

Obrigado por contribuir com o projeto **AI Model Runtime**.
Este documento descreve o fluxo de trabalho, regras e padrões que devem ser seguidos para manter a qualidade e rastreabilidade do desenvolvimento.

---

# 1. Fluxo de Trabalho Git

### Branch principal
  - **main** é protegida:
  - Não permite commits diretos.
  - Exige *merge request* + aprovação de revisores autorizados.

### Desenvolvimento em branches secundárias
- Todo desenvolvimento deve ocorrer em branches próprias.
- Nome da branch: que tenha relação com o que está sendo desenvolvido

Exemplos:
- `configuracao-ci-card-1`
- `melhorias-docs-card-3`

---

# 2. Convenção de Commits

- Os commits devem ser claros e descritivos.
- Recomendado usar *Commitizen* com mensagens no formato:

type: descrição curta

- Exemplos:
- `feat: adicionar validação de entrada`
- `fix: corrigir erro de inicialização`
- `docs: atualizar instruções de uso`

---

# 3. Merge Requests

### Todo MR deve:
- Ser criado apontando para a branch **main**.
- Passar pelo pipeline CI/CD.
- Ser revisado antes do merge.
- Seguir a descrição mínima:
- O que foi implementado
- Como testar
- Possíveis impactos

---

# 4. Kanban Técnico

Um Kanban digital foi criado exclusivamente para tarefas técnicas internas. 

### Colunas:
- **Backlog**
- **To do**
- **Doing**
- **Done**

### Tags disponíveis:
- Finished
- infra
- Documentation
- MVP
- Action needed 
- Later 
- To review 

Este Kanban **não substitui** o Kanban geral do Scrum. 
Ele complementa a organização das tarefas de desenvolvimento técnico.

---

# 5. .gitignore preparado para:
- **LaTeX**
- **Python**
- **C++**

(Arquivo já incluído no repositório.)

---

# 6. Pipeline inicial de CI/CD

O pipeline básico está configurado no GitLab CI que gera uma mensagem "Pipeline CI/CD inicial funcionando":

```yaml stages:
  - sanity

sanity_check:
  stage: sanity
  tags:
    - ai-model-runtime
  script:
    - echo "Pipeline CI/CD inicial funcionando"
  rules:
    - if: '$CI_MERGE_REQUEST_TARGET_BRANCH_NAME == "main"'
      when: always
```

