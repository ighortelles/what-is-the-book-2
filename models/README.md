# Modelos para apresentação

Os checkpoints estão incluídos no repositório (cerca de 13 MB no total).
Não é necessário baixar o dataset, montar o Drive ou repetir treinamentos.

| Pasta | Modelo | Entrada | mAP50–95 no teste |
| --- | --- | --- | --- |
| `exp_05` | YOLOv8n sem filtro adicional | 960 pixels | 76,76% |
| `preproc__bilateral_5` | YOLOv8n com Bilateral | 640 pixels | 65,70% |

Cada pasta contém `best.pt` e `result.json`, com os parâmetros necessários à
inferência. O filtro Bilateral (d=5, sigmaColor=50, sigmaSpace=50) é aplicado
automaticamente pelo app, antes da previsão. As caixas aparecem na foto original.

`ranking.json` preserva os escores de teste dos dois modelos selecionados;
`manifest.json` registra tamanho e SHA-256 dos pesos copiados sem modificação.
São cópias dos checkpoints treinados pelo projeto. As diferenças de treinamento
impedem interpretar a diferença de mAP como efeito isolado do filtro.

Somente esses dois checkpoints foram liberados no `.gitignore`; outros pesos,
datasets, ambientes e resultados de treinamento continuam ignorados.
Os pesos usam a arquitetura e as dependências Ultralytics do projeto.
