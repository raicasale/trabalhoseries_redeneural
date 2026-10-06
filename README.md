# Trabalho de Séries Temporais

Relatórios e materiais do Grupo 4 (especialização em MLP Regressor), usando a
base Daily Delhi Climate do Grupo 1.

## Documentação

- [Relatório Delhi Temp](relatorios/relatorio_tecnico_delhi_temp.html):
  versão executiva, responsiva e pronta para impressão.
- [Relatório técnico em PDF](relatorios/relatorio_tecnico_delhi_temp.pdf):
  cópia gerada a partir do HTML, com figuras embutidas.
- [Documentação completa do treinamento](docs/DOCUMENTACAO_TREINAMENTO.md):
  metodologia, auditorias, tabelas completas e instruções de reprodução.
- [Relatório Bike Sales](relatorios/relatorio_tecnico_bikes_sales.html):
  referência visual original do projeto, mantida no repositório.
- [Design system](design-system/stripe-DESIGN.md): tokens de cor, tipografia,
  espaçamento e componentes usados nos relatórios.

## Estrutura dos relatórios

```text
relatorios/
  relatorio_tecnico_delhi_temp.html
relatorio_tecnico_delhi_temp.pdf
  relatorio_tecnico_bikes_sales.html
design-system/
  stripe-DESIGN.md
```

Para reproduzir: `pip install -r requirements.txt`, execute os testes,
`src/pipeline_delhi.py`, `src/build_notebooks_delhi.py`,
`src/build_docs_delhi.py` e `src/build_report_delhi.py`. O relatório HTML é
autocontido, com figuras embutidas, e pode ser salvo como PDF pelo botão de
impressão.
