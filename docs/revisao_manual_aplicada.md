# Revisão manual aplicada

Em 14/11/2026 foram aplicadas as decisões de
`data/annotation_review/queue_reviewed.csv`, após autorização do usuário.
SHA-256 do CSV: `bd54fe751d009092308730e61561f9929f938c95935db97e75de1db52ee2cda5`.

| Divisão | Antes | Descartadas | Atual |
|---|---:|---:|---:|
| Treino | 1.599 | 60 | 1.539 |
| Validação | 400 | 12 | 388 |
| Teste | 500 | 11 | 489 |
| **Total** | **2.499** | **83** | **2.416** |

Restam 46.089 caixas; os descartes retiraram 1.078 caixas do dataset ativo.
As nove imagens marcadas para manter e as imagens sem decisão de descarte
permaneceram. As 91 imagens sinalizadas tinham decisão registrada; uma imagem
adicional também foi revisada. Registros não revisados não foram descartados.

## Procedimento e rastreabilidade

- Script: `src/preprocess/apply_review.py`.
- Verificados IDs únicos, origem do caminho, divisão e consistência das decisões.
- Somente registros com `decision=excluir` e `discard=sim` foram retirados.
- Imagens e respectivos TXT foram movidos para
  `data/discarded/manual_review/files/`, preservando seus bytes e caminhos por divisão.
- Os metadados anteriores, CSV e plano de movimentação com hashes foram
  arquivados em `data/discarded/manual_review/`. Os datasets brutos não mudaram.
- Manifesto e resumo foram atualizados. `data/unified/manual_review.json` registra
  IDs descartados, contagens e hash da revisão. Uma cópia do CSV está em
  `data/unified/provenance/manual_review.csv`.
- Os caminhos e a classe do YAML permanecem válidos. Não houve nova divisão,
  mudança de caixas ou redistribuição de imagens entre conjuntos.
- A revisão humana alterou também o teste: 11 imagens foram retiradas antes de
  qualquer treinamento. Essa mudança decorre da revisão de qualidade, sem uso
  de previsões ou métricas do modelo. O novo teste contém 489 imagens.

## Verificação e recuperação

Ruff e Pyright aprovados. `verify_dataset.py` conferiu hashes das imagens ativas
contra o inventário original, pares imagem/anotação, caixas, contagens e ausência
de grupos compartilhados. Também verificou os hashes dos arquivos arquivados e
que as mudanças correspondem exatamente à lista autorizada pelo CSV.

O script de aplicação recusa revisão diferente se uma já estiver aplicada e
recusa prosseguir diante de arquivo de recuperação de uma execução interrompida.
Repetir com o mesmo CSV é uma operação sem alterações.

Para recuperação manual, o plano `move_plan.json` registra cada origem e destino;
os arquivos e metadados anteriores estão no arquivo de recuperação. Não executar
uma restauração parcial sem recompor conjuntamente manifesto, resumo e registros
da revisão. Nenhum arquivo foi apagado definitivamente.

`data/colab_bundle.zip` foi regenerado com o dataset atual. O pacote não contém
as imagens descartadas. Os relatórios e prévias da auditoria inicial são
históricos; o manifesto atual é a referência para treinamento. O HTML de revisão
mantém acesso às imagens arquivadas para consulta das decisões anteriores.
