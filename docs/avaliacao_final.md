# Avaliação final e ranking de todos os modelos

`notebooks/yolov8_final_test_colab.ipynb` avalia a grade original e as variantes
com preprocessamento. Os treinamentos calculam apenas validação; o notebook usa
`split='test'`, sem treinar ou modificar checkpoints.

## Execução

O padrão é `MODE='local'`, usando o ambiente instalado em
`.venv`. Abra no VS Code/Jupyter e execute as células em
ordem. A inferência ocorre em subprocesso desse ambiente, sem alterar CUDA.
`DEVICE='0'` usa GPU; `DEVICE='cpu'` permite CPU. Não concorra com treinamento
ou outra avaliação na mesma GPU.

Caminhos locais padrão:

- `GRIDS`: `runs\grid` e `models`.
- `PREPROCESSING_GRID`: `runs\preprocessing_local\grid`.
- `OUTPUT`: `runs\final_test_all`.
- `CACHE`: `evaluation_test_cache`.

Nesta atualização estão disponíveis localmente as 11 variantes e o exp_05
importado. Outros originais só entram quando seus arquivos estiverem acessíveis
nos caminhos configurados. O notebook exibe a cobertura e não baixa modelos.
Aceita `exp_XX/result.json` mais `exp_XX/weights/best.pt` ou `exp_XX/best.pt`.
Várias pastas `GRIDS` são deduplicadas pelo identificador; checkpoints conflitantes
geram erro. Variantes usam `<variante>/result.json` e `<variante>/weights/best.pt`.
Modelos incompletos são pulados; precisa haver ao menos um concluído. Não use smoke.

No Colab, escolha `MODE='colab'`, disponibilize ambas as grades no Drive e confira
os caminhos. A preparação monta o Drive, solicita o ZIP caso falte o dataset e
instala o lockfile. Sem GPU, use CPU. O notebook contém uma cópia do avaliador,
mas exige também `src/preprocess/image_variants.py` no projeto extraído.
O Colab não acessa os arquivos locais do computador.

## Integridade e preprocessamento

O avaliador verifica grupos separados entre partições, arquivos de teste e
identidade do dataset de treinamento. Para a grade original, verifica o hash do
manifesto; para variantes, os hashes dos arquivos de treino/validação e do código
de preprocessamento, além das versões OpenCV/NumPy. Exige a mesma versão
Ultralytics do treinamento. O teste revisado tem 489 imagens.

Cada variante usa as mesmas funções `prepare_variant`, `read_image` e `transform`
do treinamento. Gera somente PNGs de teste, preservando geometria e bytes das
anotações. Inclusive `preproc__original` passa pela exportação PNG de seu treino.
O YAML temporário aponta os campos exigidos para `test/images`; somente
`model.val(split='test')` é chamado. Não modifica os arquivos fonte e remove o
cache de cada variante após salvar suas métricas. Cache incompleto é refeito.

Usa a resolução de treinamento de cada modelo, batch 1 por padrão, FP32, sem
aumentos, `conf=0.001`, IoU de NMS `0.7` e `max_det=1000`. Precisão e recall são
os retornados pelo validador; o limiar baixo serve para curvas AP, não recomenda
um limiar de implantação. Tempos do validador não incluem geração offline de filtros.

## Ranking

Antes da inferência, seleciona pelo mAP50–95 de validação entre os disponíveis e
registra em `evaluation_plan.json`. Depois ordena o ranking de teste por mAP50–95
decrescente, com desempate pelo identificador. A escolha por validação permanece
indicada separadamente. O ranking é descritivo; não reutilize o teste para ajustar
hiperparâmetros. As grades têm batches e aumentos diferentes: para isolar o efeito
dos filtros, compare as variantes locais com `preproc__original`.

Arquivos em `OUTPUT`:

- `evaluation_plan.json`: seleção, parâmetros, versões e hashes.
- `skipped_experiments.json`: modelos incompletos ou indisponíveis.
- `comparison.csv/json`: resultados por modelo, família, filtro e configurações.
- `ranking.csv/json`: classificação final por mAP50–95 de teste, com coluna `rank`.
- `<modelo>/test_metrics.json`: métricas e tempos.
- `<modelo>/plots`: curvas, matriz de confusão e exemplos.
- `evaluation_*.log`: log completo do subprocesso.

Reexecuções idênticas reutilizam avaliações concluídas. Mudanças nos modelos,
dados, código ou configuração exigem nova pasta `OUTPUT`, preservando avaliações
anteriores. O ranking final só é publicado quando todos os modelos disponíveis
terminam; a comparação é salva progressivamente.

Testes automatizados verificam seleção por validação, `split='test'`, filtro real
em imagem sintética, preservação das anotações, limpeza do cache, exportações
compactas do Drive, ranking e reutilização. A atualização do notebook não executa
por si só a avaliação real dos checkpoints.

Os caminhos locais são relativos ao armazenamento configurado, cujo padrão é
a raiz do repositório. Consulte [configuração de caminhos](caminhos.md)
para usar outro disco sem alterar o código.
