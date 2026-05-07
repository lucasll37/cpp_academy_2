<div align="center">

  <h1>
    <a href="https://www.asa.dcta.mil.br">
      <img
        src="./docs/logo/miia-1024x768.png"
        alt="MIIA"
        width="500"
      >
    </a>
    <br>
    MIIA
  </h1>

  <h4>
    Mecanismos de Interoperabilidade entre Modelos de Inteligência Artificial e
    Agentes Autônomos em Simuladores Construtivos.
  </h4>

</div>

<div align="center">
  <a href="https://gitlab.asa.dcta.mil.br/asa/miia/-/pipelines">
    <img src="https://gitlab.asa.dcta.mil.br/asa/miia/badges/main/pipeline.svg" alt="pipeline status">
  </a>
  <a href="https://gitlab.asa.dcta.mil.br/asa/miia/-/pipelines">
    <img src="https://gitlab.asa.dcta.mil.br/asa/miia/badges/main/coverage.svg" alt="coverage report">
  </a>
  <a href="https://gitlab.asa.dcta.mil.br/asa/miia/-/releases">
    <img src="https://gitlab.asa.dcta.mil.br/asa/miia/-/badges/release.svg" alt="Latest Release">
  </a>
</div>

<div align="center">
  <a href="#principais-features">Principais Features</a> •
  <a href="#documentacao">Documentação</a> •
  <a href="#licenca">Licença</a>
</div>

## Principais Features
* Interoperabilidade controlada entre modelos de IA e agentes autônomos no ciclo do simulador construtivo.
* Suporte a múltiplos formatos de exportacão de modelos (ex.: PyTorch, ONNX).
* Mecanismos de rastreabilidade, versionamento e metadados dos modelos de IA.
* Instrumentacão e monitoramento da inferência dos modelos durante a simulação.
* Pipeline de validacão mínima de modelos (integridade, compatibilidade e controle de seeds).

## Documentação

A documentação do projeto é gerada automaticamente a partir do código-fonte via [Doxygen](https://www.doxygen.nl).
Ela cobre tanto a API interna voltada a **desenvolvedores** quanto os guias de uso destinados a **usuários** da ferramenta.

**Pré-requisitos:** `doxygen`, `graphviz` e `make` instalados.

Instalar dependências (Ubuntu/Debian):

```bash
sudo apt update && sudo apt install -y make graphviz doxygen
```

Gerar a documentação:

```bash
make docs
```

Abrir documentação no navegador:

```bash
make docs-open
```

## Licença

Este projeto é de uso institucional restrito e protegido pela legislação brasileira de software.

Consulte o arquivo [LICENSE.md](LICENSE.md) para os termos completos de uso, conforme a Lei nº 9.609/1998.

<!-- <div align="center">

  <h1>
    <a href="https://www.asa.dcta.mil.br">
      <img
        src="./docs/images/logo/miia-1024x768.png"
        alt="MIIA"
        width="500"
      >
    </a>
    <br>
    MIIA
  </h1>

  <h4>
    Mecanismos de Interoperabilidade entre Modelos de Inteligência Artificial e
    Agentes Autônomos em Simuladores Construtivos.
  </h4>

</div>

<div align="center">
  <a href="https://gitlab.asa.dcta.mil.br/asa/miia/-/pipelines">
    <img src="https://gitlab.asa.dcta.mil.br/asa/miia/badges/main/pipeline.svg" alt="pipeline status">
  </a>
  <a href="https://gitlab.asa.dcta.mil.br/asa/miia/-/pipelines">
    <img src="https://gitlab.asa.dcta.mil.br/asa/miia/badges/main/coverage.svg" alt="coverage report">
  </a>
  <a href="https://gitlab.asa.dcta.mil.br/asa/miia/-/releases">
    <img src="https://gitlab.asa.dcta.mil.br/asa/miia/-/badges/release.svg" alt="Latest Release">
  </a>
</div>

<div align="center">
  <a href="#principais-features">Principais Features</a> •
  <a href="#licenca">Licença</a>
</div>

## Principais Features
* Interoperabilidade controlada entre modelos de IA e agentes autônomos no ciclo do simulador construtivo.
* Suporte a múltiplos formatos de exportacão de modelos (ex.: PyTorch, ONNX).
* Mecanismos de rastreabilidade, versionamento e metadados dos modelos de IA.
* Instrumentacão e monitoramento da inferência dos modelos durante a simulação.
* Pipeline de validacão mínima de modelos (integridade, compatibilidade e controle de seeds).

## Licença

Este projeto é de uso institucional restrito e protegido pela legislação brasileira de software.

Consulte o arquivo [LICENSE.md](LICENSE.md) para os termos completos de uso, conforme a Lei nº 9.609/1998. -->
