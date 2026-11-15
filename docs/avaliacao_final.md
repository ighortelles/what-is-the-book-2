# Avaliação final dos experimentos sem preprocessamento

O treinamento original (`train_grid.py` e `yolov8_colab.ipynb`) calcula métricas
somente em `val`. Seus arquivos `result.json` e `ranking.json` não são resultados
de teste. Não foi possível verificar eventuais avaliações avulsas no Drive.

Abra `notebooks/yolov8_final_test_colab.ipynb` no Colab e execute em ordem:

1. Envie o mesmo `colab_bundle.zip` com o dataset revisado.
2. Instale as dependências e monte o Drive.
3. Confira `GRID`, apontando para `MyDrive/what-is-the-book-2/runs/grid`.
4. Execute a avaliação e visualize os relatórios.

CPU é a opção padrão, permitindo executar mesmo sem cota de GPU. Caso disponha
de GPU no Colab, use `DEVICE='0'`. Não é necessário baixar os resultados do Drive
para o computador. O notebook incorpora o avaliador para funcionar com o ZIP já
existente. Ele não modifica checkpoints nem inicia treinamentos.

São avaliados os experimentos que possuem `result.json` e `weights/best.pt`.
Experimentos incompletos são pulados, com os arquivos ausentes registrados no log
e em `skipped_experiments.json`. É necessário pelo menos um experimento concluído.
A seleção por validação considera somente os modelos disponíveis nesta execução.
Para incluir experimentos finalizados posteriormente, escolha uma nova pasta
`OUTPUT`, preservando o plano e os resultados anteriores. A grade de preprocessamento local
e sua referência `original` são experimentos separados e não entram nesta tabela.

O avaliador confere o hash do manifesto usado no treinamento, a separação dos
grupos entre partições e a correspondência dos arquivos de teste com o manifesto.
Também registra hashes dos pixels codificados e anotações de teste, dos checkpoints
e do próprio avaliador, além das versões e parâmetros. O hash novo do teste
documenta os arquivos avaliados; o treinamento antigo registrava apenas o manifesto,
portanto esse registro não prova que os bytes das imagens nunca foram alterados.

A escolha do modelo principal usa exclusivamente o mAP50–95 de validação e é
salva antes de executar inferência no teste. A comparação dos oito modelos no
teste é descritiva: não use esses resultados para selecionar hiperparâmetros ou
ajustar limiares. O conjunto revisado contém 489 imagens de teste; a contagem
efetiva é conferida no manifesto e exibida pelo notebook.

Cada modelo usa a resolução de seu treinamento. Batch 1, precisão FP32, sem
aumento de teste; `conf=0.001`, IoU de NMS `0.7`, `max_det=1000`.
Não há filtros ou operadores de borda, apenas o processamento padrão do YOLO.
Precisão e recall são os valores retornados pelo validador, não métricas de um
limiar de confiança escolhido para implantação. Latências são registradas por
imagem e dependem do dispositivo; não são medidas de tempo total de aplicação.

Os arquivos ficam em `runs/final_test_original` no Drive:

- `evaluation_plan.json`: seleção por validação, parâmetros, hashes e experimentos pulados.
- `skipped_experiments.json`: experimentos incompletos e seus arquivos ausentes.
- `comparison.csv` e `comparison.json`: métricas de validação e teste por experimento.
- `exp_XX/test_metrics.json`: precisão, recall, mAP50, mAP50–95 e tempos.
- `exp_XX/plots`: curvas, matriz de confusão e exemplos do validador.
- `evaluation_*.log`: logs completos sem sobrecarregar a saída do notebook.

Ao repetir a execução com a mesma configuração, resultados completos são
reutilizados. Um modelo interrompido é reavaliado. Mudanças na configuração exigem
uma nova pasta de saída. Não execute duas avaliações simultâneas na mesma pasta.

O programa também está em `src/detection/evaluate_grid.py`, com opções
`--data`, `--grid`, `--output`, `--device` e `--batch`.
Os testes automatizados usam modelos simulados para verificar a escolha pela
validação, o uso exclusivo de `split='test'`, a reutilização dos resultados e
o salto de experimentos incompletos, a rejeição de uma grade sem modelos concluídos
e a rejeição de grupos compartilhados entre partições.
Não foi executada avaliação real nesta alteração; os checkpoints originais
permanecem no Drive, e o treinamento local continua independente.
