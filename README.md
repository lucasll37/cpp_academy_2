<p align="center">
  <img src="./docs/img/asa2-miia-1019x117.png" alt="ASA2-MIIA" />
</p>

<div align="center">
  [![pipeline status](https://gitlab.asa.dcta.mil.br/asa/asa2-miia/badges/main/pipeline.svg)]( https://gitlab.asa.dcta.mil.br/asa/asa2-miia/-/pipelines)
  [![coverage report](https://gitlab.asa.dcta.mil.br/asa/asa2-miia/badges/main/coverage.svg)]( https://gitlab.asa.dcta.mil.br/asa/asa2-miia/-/pipelines)
  [![Latest Release](https://gitlab.asa.dcta.mil.br/asa/asa2-miia/-/badges/release.svg)](https://gitlab.asa.dcta.mil.br/asa/asa2-miia/-/releases)
</div>

---

# AI Model Runtime

Repositório oficial do projeto **AI Model Runtime**, organizado segundo as diretrizes internas do ASA e integrado ao fluxo de desenvolvimento GitLab + Nextcloud.

Este documento apresenta a visão geral da estrutura do projeto, sua organização, regras de contribuição e funcionamento do pipeline básico de CI/CD.

## Estrutura e Padrões do Repositório

### Estrutura no Nextcloud

Toda documentação e organização administrativa está no diretório raiz **ASA2** no Nextcloud:
```yaml stages:
ASA2
│
├── Apresentações
│
├── Formulários Finais para Assinar
│
├── Pesquisa
│ └── Mecanismos de Integração para Execução de Modelos de Aprendizado de Máquina
│ 	├── admin
│ 	├── docs
│ 	│ ├── artigos (contém artigos pesquisados durante a execução dos Cards)
│ 	│ └── Cards (São documentos com descrição de tarefas micro para a conclusão do Scrum geral)
│ 	├── mvp
│ 	└── research
│
├── Revisões
│
├── Sprints
│ 	├── 0
│ 	└── exemplos
│
└── Técnico
```

### Estrutura Inicial do Repositório

A estrutura base do repositório segue o padrão definido:

```yaml stages:
ai-model-runtime/
│
├── artifacts
├── build
├── ci
├── deploy
├── docs
├── examples
├── include
├── scripts
├── src
├── tests
├── third_party
└── tools
```