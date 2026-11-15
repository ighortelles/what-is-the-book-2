# what-is-the-book-2

Trabalho II de Visão Computacional: detecção de lombadas com caixas retangulares.

## Estrutura

```text
src/preprocess/         Scripts de auditoria e tratamento de dados
docs/                  Enunciado e documentação metodológica
data/raw/              Datasets extraídos, preservados
data/zips/             Downloads originais, preservados
data/audit/            Inventário, métricas de similaridade e prévias
data/unified/          Dataset YOLO unificado, manifesto e proveniência
```

Novos scripts de tratamento de dados devem ficar em `src/preprocess/`.
O histórico dos tratamentos, critérios de deduplicação, limitações e resultados
está em [Dataset unificado](docs/dataset_unificado.md).

## Ambiente e execução

Requer Python 3.13 e uv. Execute na raiz deste projeto:

```powershell
uv sync --locked --group dev
uv run python src/preprocess/prepare_dataset.py --help
uv run python src/preprocess/prepare_dataset.py audit
uv run python src/preprocess/prepare_dataset.py build
```

`audit` gera o inventário a partir dos dados originais. `build` exige esse
inventário e recusa sobrescrever `data/unified/` se ela já existir.
Após a revisão manual, o dataset tem **1.539 imagens de treino, 388 de validação
e 489 de teste** (2.416 imagens e 46.089 caixas). O teste permanece reservado à
avaliação final. Para reproduzir desde
os dados brutos, execute `audit`, `build` e então
`uv run python src/preprocess/create_validation.py`. Não repita essa divisão
na saída atual: o script recusa dividir novamente.

As decisões de `data/annotation_review/queue_reviewed.csv` foram aplicadas com
`uv run python src/preprocess/apply_review.py`. Os 83 descartes estão preservados
em `data/discarded/manual_review/`, fora do dataset e do pacote Colab. A aplicação
não altera anotações nem redistribui imagens; sua execução repetida com o mesmo
CSV não faz novas alterações. Execute `uv run python src/preprocess/verify_dataset.py`
para verificar a integridade. Detalhes em [Revisão manual aplicada](docs/revisao_manual_aplicada.md).

## Revisão e experimentos YOLOv8

Consulte [Revisão e experimentos](docs/revisao_e_experimentos.md).

```powershell
uv run python src/preprocess/review_annotations.py
uv run python src/detection/train_grid.py
uv run python src/preprocess/package_colab.py
```

Abra `data/annotation_review/index.html` para conferir as imagens e registre
decisões em `queue.csv` na mesma pasta. Preserve uma cópia preenchida antes de
regenerar a fila, pois o script a recria. Corrigir anotações exige atualização
consistente do manifesto e nova auditoria; a página apenas visualiza os dados.

O segundo comando apresenta a grade, sem treinar. O terceiro prepara
`data/colab_bundle.zip`. Abra [o notebook](notebooks/yolov8_colab.ipynb) no Google
Colab com GPU e siga suas células. O ambiente de treinamento usa a dependência
opcional `training`, instalada no Colab com `uv sync --locked --extra training`.
Os treinamentos não foram executados nesta preparação.

## Experimentos com filtros e bordas

O notebook [yolov8_preprocessing_colab.ipynb](notebooks/yolov8_preprocessing_colab.ipynb)
compara 11 variantes: RGB, cinza, Gaussianos 3×3/5×5, mediana, bilateral,
Canny com dois pares de limiares, Sobel, Laplaciano e RGB com Sobel.
Os hiperparâmetros são fixos e os aumentos aleatórios são desativados para
isolar o efeito do processamento. Treino e validação recebem a mesma operação;
o teste permanece reservado. Os originais e as anotações não são modificados.

Envie o `data/colab_bundle.zip` atualizado. Os resultados ficam em
`runs/preprocessing/`, separados da grade anterior, com retomada automática e
logs resumidos. Imagens processadas são geradas no Colab, uma variante por vez;
o cache é removido após salvar cada resultado. O protocolo está em
[experimentos_preprocessamento.md](docs/experimentos_preprocessamento.md).

Verificação local, sem treinamento ou download de pesos:

```powershell
uv run python -m unittest discover -s tests -p test_preprocessing.py
uv run python -m src.detection.train_preprocessing --preview --output data/preprocessing_preview
```

## Padronização do código

As configurações ficam em `pyproject.toml` e as versões resolvidas em `uv.lock`.

- **Ruff:** Python 3.13, linhas de até 100 caracteres, aspas duplas, espaços e
  regras `E`/`F` (erros e análise estática), `I` (imports), `UP` (sintaxe moderna)
  e `B` (padrões propensos a bugs). Notebooks permitem imports fora do topo (`E402`).
- **Pyright:** análise de `src/` em modo `basic`, imports ausentes tratados como
  erro e ambiente `.venv` local. Não exige stubs para bibliotecas sem tipagem.
- Dados, arquivos temporários e ambientes virtuais ficam fora da análise.

Verificações antes de entregar alterações:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Para aplicar formatação e correções automáticas seguras:

```powershell
uv run ruff check --fix .
uv run ruff format .
```
