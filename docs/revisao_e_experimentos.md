# Validação, revisão de anotações e YOLOv8

> Registro histórico de preparação. As menções abaixo a treinamentos ainda não
> executados e à ausência de retomada descrevem etapas anteriores. Para o estado
> atual, consulte a [avaliação final](avaliacao_final.md).

## Divisão anterior à aplicação da revisão manual

A versão atual, após os 83 descartes autorizados no CSV, tem 1.539 imagens de
treino, 388 de validação e 489 de teste. Consulte `revisao_manual_aplicada.md`.
As contagens e achados abaixo registram a auditoria anterior aos descartes.

| Origem | Treino | Validação | Teste |
|---|---:|---:|---:|
| Book spine dataset-library | 558 | 139 | 174 |
| Book spine instance segmentation | 922 | 231 | 288 |
| book | 62 | 16 | 20 |
| bookshelf-ai | 57 | 14 | 18 |
| **Total** | **1.599** | **400** | **500** |

Caixas: 29.319 no treino, 7.440 na validação e 10.408 no teste.
A validação utiliza aproximadamente 20% do treino anterior, mantendo as
proporções por origem. Semente 43, 2.000 tentativas de ordenação dos grupos,
minimizando o erro relativo quadrático da quantidade-alvo por origem.
Os grupos do manifesto são indivisíveis. Os registros e arquivos de teste
permanecem os mesmos. Não foram criadas novas imagens nem removidos objetos.

`src/preprocess/create_validation.py` move pares imagem/anotação e atualiza
manifesto, resumo e YAML. O backup `data/unified/before_validation/` preserva
os três arquivos de metadados anteriores e o plano de movimentação. O script
recusa uma segunda divisão ou um backup de execução interrompida.

## Auditoria das anotações

`src/preprocess/review_annotations.py` analisou as 2.499 imagens e 47.167 caixas.
Não foram encontradas classes incorretas, coordenadas inválidas ou áreas nulas.
Os sinais abaixo são heurísticos; uma imagem pode ter mais de um sinal.

| Sinal | Imagens | Critério |
|---|---:|---|
| Sobreposição forte | 24 | Pelo menos duas caixas com IoU > 0,85 |
| Objeto muito pequeno | 3 | Lado menor < 3 pixels após escala proporcional para lado maior 640 |
| Anotação vazia | 41 | Nenhuma caixa; verificar se é realmente imagem negativa |
| Diferença entre cópias | 23 | Representante de duplicatas com valores de caixas diferentes |
| **União dos sinais** | **91** | Imagens distintas |

Foi inspecionada uma amostra aleatória de 24 imagens (duas por origem e divisão,
semente 42) e uma seleção dirigida de seis casos, três deles já na amostra.
Isso totaliza 27 imagens distintas revisadas visualmente. Não equivale à revisão
manual integral de todas as caixas. Não houve correção automática de rótulos.

### Achados que precisam de revisão manual

- **ID 251:** possíveis lombadas sem caixa à direita do conjunto principal.
  Confirmar quais objetos são lombadas e completar as anotações se necessário.
- **IDs 100 e 177:** conferir limites das caixas à direita, livros estreitos e
  a política para livros parcialmente visíveis nas prateleiras adjacentes.
- **IDs 209, 222 e 285:** conferir caixas muito sobrepostas. A conversão de
  polígonos inclinados em retângulos pode produzir sobreposição legítima;
  não apagar uma caixa apenas por ter IoU elevado.
- **IDs 1035 e 2097:** a amostra visual mostra fundo/prateleira sem livros
  identificáveis; anotações vazias parecem adequadas nesses dois casos. Os
  outros negativos ainda precisam de conferência.
- **Diferenças entre cópias:** revisar os 23 representantes. A seleção anterior
  por maior número de caixas não garante que a anotação escolhida seja a melhor.

Abra `data/annotation_review/index.html` no navegador. É uma página local,
com filtro por sinal, amostra por origem/divisão, navegação e opção de ocultar
as caixas. Não requer servidor, não envia imagens e não altera as anotações.
`queue.csv` contém IDs, origem, divisão, caminho, sinais e campos vazios para
`decision` e `notes`. Sugestões de decisão: `ok`, `corrigir`, `excluir`, `duvida`.
Guarde uma cópia preenchida antes de regenerar a auditoria: a fila é sobrescrita.

As imagens do teste podem ser revisadas para qualidade dos rótulos antes do
experimento. A política de anotação deve ser a mesma nos três conjuntos, sem
usar previsões do modelo ou métricas do teste para decidir o que corrigir.
Após revisão, congelar uma nova versão dos dados e atualizar contagens/hashes
se houver mudanças. Regenerar o pacote do Colab após qualquer correção.

## Grade inicial de treinamento

Configuração: `configs/yolov8_grid.json`. Modelo: **YOLOv8n**, pesos COCO
`yolov8n.pt`. Cada experimento reinicializa o mesmo modelo pré-treinado.

| Experimento | Resolução | Taxa inicial | Mosaic |
|---|---:|---:|---:|
| 1 | 640 | 0,001 | 0 |
| 2 | 640 | 0,001 | 1 |
| 3 | 640 | 0,0003 | 0 |
| 4 | 640 | 0,0003 | 1 |
| 5 | 960 | 0,001 | 0 |
| 6 | 960 | 0,001 | 1 |
| 7 | 960 | 0,0003 | 0 |
| 8 | 960 | 0,0003 | 1 |

Fixos: AdamW, batch 8, máximo de 50 épocas, patience 15, weight decay 0,0005,
semente 42, execução determinística, AMP, dois workers e max_det 1.000.
Mosaic é desligado nas dez últimas épocas quando ativado. Flips, rotações
adicionais e mixup estão desativados; outros parâmetros seguem os defaults da
versão travada da Ultralytics e são registrados em `args.yaml` de cada execução.
O limite elevado de detecções evita impor um teto baixo em estantes densas.

O otimizador é explícito: `optimizer=auto` pode ignorar o `lr0` fornecido,
inviabilizando essa comparação. Referências oficiais:
[treinamento](https://docs.ultralytics.com/modes/train/) e
[YOLOv8](https://docs.ultralytics.com/models/yolov8/).

PyTorch 2.7.1 e torchvision 0.22.1 foram fixados com Python 3.13. O lock
resolve a distribuição Linux com CUDA 12.6. Essa combinação é documentada em
[versões do PyTorch](https://pytorch.org/get-started/previous-versions/).
O notebook verifica disponibilidade da GPU antes de treinar; a sessão real do
Colab ainda precisa executar a verificação curta para confirmar compatibilidade.

## Execução no Colab e avaliação

1. Revisar os casos manuais e registrar decisões.
2. Gerar `data/colab_bundle.zip` com `src/preprocess/package_colab.py`.
3. Abrir `notebooks/yolov8_colab.ipynb` no Colab, selecionar GPU, enviar o ZIP e
   executar a instalação isolada com uv/Python 3.13. O notebook monta o Drive
   para persistir os resultados, sujeito à autorização na interface do Google.
4. Rodar o smoke test (uma época e 5% do treino). Não usar suas métricas como
   resultado. Se faltar memória, reduzir batch para toda a grade e documentar.
5. Executar os experimentos individualmente, preservando as mesmas condições.
   O script permite toda a grade, mas uma sessão pode não durar o suficiente.
6. Comparar mAP50–95 de validação em `ranking.json`, junto com precisão,
   recall, mAP50, curvas e exemplos. O checkpoint é selecionado pela lógica de
   fitness da Ultralytics; o ranking utiliza a avaliação explícita de `best.pt`
   na validação. Apenas execuções concluídas entram no ranking.
7. Após escolher a configuração, considerar repetições com outras sementes
   antes da avaliação final no teste. Essa etapa ainda não está implementada.

Cada execução salva parâmetros solicitados, versões, hash do manifesto,
`args.yaml`, `results.csv`, gráficos, pesos e resultado de validação. Pastas
existentes não são sobrescritas. Não há retomada automática de execução
interrompida: preservar o diretório e usar outra pasta para refazer.

**Estado:** dados divididos e revisados por heurísticas/amostragem; grade e
notebook preparados. Nenhum treinamento ou acesso ao Colab foi executado.
A execução local em modo de planejamento não importa PyTorch/Ultralytics,
permitindo validar a grade sem instalar dependências pesadas no computador.

## Verificações executadas

### Correção do backend de gráficos no Colab

O teste no Colab reconheceu a Tesla T4, mas falhou antes de treinar porque o
subprocesso herdou `MPLBACKEND=module://matplotlib_inline.backend_inline` do
notebook. Esse backend não estava disponível no ambiente isolado. O script
agora define `MPLBACKEND=Agg` antes de importar Ultralytics, permitindo salvar
gráficos sem depender da interface do notebook. Na sessão já aberta, definir
`os.environ["MPLBACKEND"] = "Agg"` antes de executar o subprocesso tem o mesmo
efeito. Repetir o teste em uma pasta nova preserva os logs da tentativa anterior.
O treinamento completo ainda depende da confirmação de uma nova execução no Colab.

- Ruff: lint e formatação aprovados; Pyright: zero erros e avisos.
- Planejamento da grade executado sem treinamento.
- `src/preprocess/verify_dataset.py`: 2.499 imagens com SHA-256 idêntico ao
  inventário de origem; contagens e coordenadas de rótulos conferidas; nenhum
  grupo compartilhado entre divisões; registros de teste inalterados.
- Pacote ZIP criado localmente. O notebook ainda não foi executado em uma GPU
  do Colab; a verificação curta é necessária antes dos experimentos completos.
