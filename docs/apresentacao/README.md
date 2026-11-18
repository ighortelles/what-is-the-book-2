# Apresentação em HTML

Abra `index.html` no Firefox, Chrome ou Edge, inclusive no Linux Mint.
É um arquivo único, sem fontes externas, bibliotecas remotas ou instalação.
Não depende da pasta `artigo` e pode ser copiado separadamente.

- Setas, Page Up/Page Down ou botões: navegação.
- Home/End: primeiro/último slide.
- F ou botão: tela cheia. F11 é a alternativa do navegador.
- O endereço `index.html#9` abre diretamente o slide de resultados principais.
- O link de demonstração requer o app em execução (`bash start_app.sh`).

## Conteúdo e evidências

13 slides: proposta, origem e preparo dos dados, exemplos anotados, treino sem filtros, treino
com filtros, medidas, resultados e comparação com o trabalho 1.

Os números foram conferidos nos registros locais dos experimentos, também
documentados em `docs/dataset_unificado.md`, `docs/revisao_manual_aplicada.md`
e no dossiê local do artigo. Os dois checkpoints apresentados estão em
`models/`, com escores em `models/ranking.json`.

O trabalho 1 usa o registro final `docs/results/test_evaluation_001.json`,
de 06/10/2026, do repositório `what-is-the-book`. São dados de teste,
não os resultados anteriores de validação. A apresentação não combina os
dois trabalhos em um ranking, nem compara F1 diretamente com mAP.

As quatro fontes de dados estão vinculadas no slide 3. A seleção de exp_05
entre oito modelos foi confirmada pelo autor. Na comparação local há dez
transformações e um controle RGB sem filtro. O desenho de livros do slide 2
é esquemático e está identificado como ilustração, não previsão real.

O slide 5 inclui a montagem original `data/annotation_review/sample_1.jpg`
(o arquivo disponível é JPG, não PNG), incorporada ao HTML para continuar
funcionando sem arquivos adicionais. A montagem antecede os descartes manuais
e inclui imagens de treino, validação e teste. As caixas são anotações,
não previsões. As fontes de dados estão creditadas no slide 3.
