# Dataset unificado de lombadas

Preparado em 14/11/2026, sem modificar `data/raw/` ou `data/zips/`.
Saída: `data/unified/`. Classe única: `0 = book` (lombada).

## Fontes dos datasets

Os quatro datasets foram obtidos no Roboflow Universe, nas versões abaixo,
com exportação YOLOv8. As licenças são as declaradas nos metadados dos exports.

| Dataset | Versão | Licença | Link |
|---|---:|---|---|
| Book spine dataset-library | 3 | CC BY 4.0 | [Acessar dataset](https://universe.roboflow.com/cnu-hmlab-syko-and-ash/book-spine-dataset-library/dataset/3) |
| Book spine instance segmentation | 1 | CC BY 4.0 | [Acessar dataset](https://universe.roboflow.com/harald-varner-xv5u7/book-spine-instance-segmentation/dataset/1) |
| book | 1 | CC BY 4.0 | [Acessar dataset](https://universe.roboflow.com/hahaha-qjv2y/book-gjdsd/dataset/1) |
| bookshelf-ai | 2 | CC BY 4.0 | [Acessar dataset](https://universe.roboflow.com/bookshelf-ai/bookshelf-ai-gopqn/dataset/2) |

Os arquivos originais de metadados e atribuição estão preservados localmente em
`data/unified/provenance/`. Como `data/` é ignorada pelo Git, esta seção mantém
os links das fontes disponíveis na documentação versionada do projeto.

## Resultado inicial (antes da separação de validação)

| Origem | Entrada | Treino | Teste | Removidas |
|---|---:|---:|---:|---:|
| Book spine dataset-library v3 | 871 | 697 | 174 | 0 |
| Book spine instance segmentation v1 | 1.463 | 1.153 | 288 | 22 |
| book v1 | 122 | 78 | 20 | 24 |
| bookshelf-ai v2 | 92 | 71 | 18 | 3 |
| **Total** | **2.548** | **1.999** | **500** | **49** |

Treino: 36.759 caixas. Teste: 10.408 caixas. Total: 47.167 caixas.
As contagens de origem referem-se à versão mantida; fotos compartilhadas entre
datasets aparecem somente sob a origem da cópia escolhida.

## Anotações e integridade

- Lidas todas as 2.548 imagens e suas anotações. Nenhum erro permaneceu.
- Convertidas 38.943 anotações poligonais da entrada em caixas alinhadas aos
  eixos, usando os extremos de cada polígono. Esse número é anterior à deduplicação.
- Preservadas caixas YOLO existentes; padronizada a classe como `book`.
- Verificados classe, valores finitos, limites normalizados e área positiva.
  Extremos de caixas foram limitados à imagem e caixas idênticas dentro de uma
  anotação foram eliminadas.
- Dois exemplos tinham caminhos longos inacessíveis pelas operações normais do
  Windows. Imagens/anotações foram lidas diretamente dos ZIPs, sem perder exemplos.
- Renomeados arquivos de saída com identificadores únicos, mantendo rastreabilidade.
- Metadados, URLs e licenças originais foram copiados para `data/unified/provenance/`.
  Os quatro exports declaram CC BY 4.0.

## Duplicatas e separação

1. Calculado SHA-256 do arquivo e dos pixels RGB com as dimensões. Não foram
   encontradas cópias com pixels exatamente iguais.
2. Comparadas todas as imagens por pHash (63 bits AC, DCT de 32 × 32).
   Distância de Hamming até 6 produziu 1.416 pares candidatos a semelhança.
3. Para remover cópias quase idênticas, exigidos simultaneamente pHash até 2,
   diferença média absoluta RGB até 3 em escala 0–255 (redimensionamento para
   128 × 128) e diferença relativa de proporção até 1%. Foram 57 pares,
   agrupados por componentes conexos, resultando em 49 arquivos removidos.
4. Em cada componente, mantida a imagem com mais caixas; empate resolvido por
   maior resolução e depois caminho em ordem alfabética. Em 34 remoções os
   valores das anotações diferiam. Isso não significa necessariamente erro:
   pequenas diferenças de coordenadas também contam. Não foram combinadas
   caixas entre imagens, evitando duplicar objetos. A escolha é heurística e
   não substitui uma revisão semântica completa das anotações.
5. Variantes com o mesmo nome anterior ao sufixo `.rf.` dentro de cada origem,
   duplicatas e todos os pares com pHash até 6 foram unidos em grupos indivisíveis.
   Após a remoção, restaram 808 grupos. Assim, variantes detectadas ficam no
   mesmo conjunto mesmo quando são preservadas por terem iluminação diferente.
6. Busca determinística com semente 42 e 2.000 tentativas de ordem dos grupos
   minimizou o desvio da meta de 20% de teste por origem. Não há grupos detectados
   nem hashes de pixels compartilhados entre treino e teste.

Foram inspecionadas visualmente amostras dos candidatos e 16 imagens de saída
com caixas, cobrindo as quatro origens e as duas divisões. A verificação visual
foi amostral. pHash e comparação de pixels não garantem detectar todo recorte,
espelhamento, rotação ou outra foto da mesma estante. A separação evita vazamento
entre as famílias detectadas; não garante independência por biblioteca ou sessão
de captura, pois esses metadados não estão disponíveis. Fotos de fundo sem caixas
são mantidas como exemplos negativos.

## Arquivos

- `data/unified/data.yaml`: caminhos de treino e teste e classe única.
- `data/unified/train/{images,labels}` e `test/{images,labels}`: dados YOLO.
- `data/unified/manifest.json`: origem, destino, divisão, grupo, hash e número
  de caixas de cada imagem mantida.
- `data/unified/duplicates.json`: IDs removidos, seus representantes e diferenças
  de anotações; IDs podem ser consultados no inventário.
- `data/unified/summary.json`: contagens por origem.
- `data/audit/inventory.json`: inventário completo, caixas, hashes, candidatos
  e métricas dos pares quase idênticos.
- `data/audit/review_*.jpg`: amostras de pares semelhantes.
- `data/audit/unified_preview.jpg`: amostra de imagens com caixas por origem/divisão.

## Reprodução

Script: `src/preprocess/prepare_dataset.py`. Requer Python com `numpy`, `Pillow` e
`opencv-python`, declarados no `pyproject.toml`. A preparação inicial foi executada
com o ambiente virtual do trabalho 1, sem alterá-lo. Na organização seguinte,
foi criado um ambiente próprio `.venv` para o trabalho 2 e um `uv.lock` com as
versões resolvidas, incluindo Ruff e Pyright como dependências de desenvolvimento.

```powershell
uv sync --locked --group dev
uv run python src/preprocess/prepare_dataset.py audit
uv run python src/preprocess/prepare_dataset.py build
```

O build recusa sobrescrever uma saída existente. Para uma nova execução, arquive
a pasta `data/unified/` antes. O YAML contém caminho absoluto para esta máquina;
ao transportar o dataset, ajuste `path`.

Na primeira etapa foram criados apenas treino e teste. Na etapa seguinte,
documentada em `revisao_e_experimentos.md`, foram separadas 400 imagens do treino
para validação. O YAML atual define `train`, `val` e `test`. Não usar o teste
para escolher hiperparâmetros ou fazer early stopping.

O diretório `data/` já é ignorado pelo Git. Este relatório e o script podem ser
versionados; as imagens e os relatórios JSON permanecem locais.

## Organização e alcance dos tratamentos

Todos os scripts de tratamento ficam em `src/preprocess/`, seguindo a organização
do trabalho 1. A raiz dos dados é resolvida a partir da localização do script,
sem depender do diretório de execução. A mudança de pasta e a padronização do
código não regeneram nem alteram o dataset já produzido.

As imagens exportadas são copiadas em bytes: não recebem redimensionamento,
recompressão, normalização de cor ou novos aumentos artificiais. Conversões RGB,
escala de cinza e redimensionamentos são usados apenas para hashes, comparação
visual e prévias. Aumentos já presentes nos datasets originais são preservados
quando não atendem ao critério de remoção, respeitando o agrupamento descrito.

O processamento utiliza os arquivos `.jpg` encontrados em `raw/`, formato de todas
as imagens desta entrada. Arquivos de anotação vazios representam exemplos sem
objetos; linhas vazias são ignoradas. Coordenadas aceitam tolerância de 0,000001
nos limites antes de serem limitadas a [0, 1]. As caixas de saída são gravadas
com nove casas decimais. Uma falha de leitura ou validação é registrada no
inventário e impede o build, em vez de excluir silenciosamente a imagem.

As divisões originais dos quatro datasets foram reunidas e substituídas pela
divisão por grupos documentada acima. Não houve balanceamento para igualar o
número de imagens de cada origem: preservou-se sua participação relativa.
O registro `original_split` no inventário mantém a divisão anterior.

A proteção contra sobrescrita, a validação de correspondência imagem/anotação,
a unicidade dos hashes e a ausência de grupos compartilhados são verificadas
pelo script. Execute Python normalmente, sem `-O`, pois essas verificações
utilizam assertions. Os comandos de Ruff e Pyright estão no README.
