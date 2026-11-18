# YOLOv8: comparação de filtros e operadores de borda

> Esta página descreve o perfil Colab planejado (batch 8). A execução local final
> utilizou batch 2 e AMP desativado. Consulte as instruções de
> [treinamento local](treinamento_local.md) e [avaliação final](avaliacao_final.md).

Notebook: `notebooks/yolov8_preprocessing_colab.ipynb`.
Configuração: `configs/yolov8_preprocessing.json`.
Preparação: `src/preprocess/image_variants.py`.
Treinamento: `src/detection/train_preprocessing.py`.

## Objetivo e controles

Comparar o efeito do processamento de imagem com YOLOv8n pré-treinado,
preservando as divisões e todas as caixas aprovadas na revisão manual.
São 11 experimentos, com os mesmos hiperparâmetros: entrada 640, AdamW,
lr0 0,001, batch 8, até 50 épocas, patience 15, weight decay 0,0005 e seed 42.
Cada variante começa com os mesmos pesos pré-treinados, sem usar o modelo
treinado pela variante anterior. Não há cruzamento com a grade anterior de
oito configurações: o fator experimental aqui é o processamento da imagem.

Mosaic, MixUp, CutMix, copy-paste, alterações HSV, espelhamentos, rotações,
translação, escala aleatória, cisalhamento e perspectiva estão desativados.
O runner recusa ambientes com Albumentations, pois sua presença pode ativar
filtros adicionais automaticamente na Ultralytics. Os demais defaults da versão
travada ficam registrados em `args.yaml`. O redimensionamento/letterbox e a
normalização para [0, 1] necessários ao YOLO continuam ativos.

A referência RGB deve ser treinada nesta grade: não comparar apenas com métricas
antigas obtidas com aumentos diferentes. A referência cinza ajuda a distinguir
o efeito de remover cores do efeito de extrair bordas.

## Variantes

| Nome | Operação e parâmetros |
|---|---|
| original | Pixels RGB originais, após decodificação |
| grayscale | Cinza replicado em três canais |
| gaussian_3 | Gaussiano 3×3, sigma 0,8 |
| gaussian_5 | Gaussiano 5×5, sigma 1,2 |
| median_3 | Mediana 3×3 |
| bilateral_5 | Bilateral: diâmetro 5, sigmaColor 50, sigmaSpace 50 |
| canny_50_150 | Canny: limiares 50/150 e norma L2 |
| canny_100_200 | Canny: limiares 100/200 e norma L2 |
| sobel | Magnitude dos gradientes X/Y, kernel 3×3 |
| laplacian | Valor absoluto do Laplaciano, kernel 3×3 |
| rgb_sobel_025 | 75% RGB + 25% magnitude Sobel em três canais |

Nos operadores de borda, converte-se primeiro para cinza e aplica-se Gaussiano
3×3 com sigma 0,8. Sobel e Laplaciano são calculados em float32 para preservar
o sinal; magnitude/valor absoluto são escalados pelo máximo da própria imagem
para [0, 255]. Imagens constantes produzem mapa zero, evitando divisão por zero.
Canny produz mapa binário. Todos os resultados têm três canais uint8, para
compatibilidade com os pesos YOLO, e mantêm exatamente altura e largura.

As operações ocorrem na resolução original, antes do redimensionamento do YOLO.
Assim, kernels fixos em pixels têm efeito relativo diferente conforme a resolução
da origem. Essa é uma limitação do protocolo inicial, não uma normalização espacial.
Mapas de borda podem perder cor/textura e destacar letras ou prateleiras; não há
garantia de melhoria. RGB com Sobel é uma mistura ponderada, não canais adicionais.

## Dados, armazenamento e retomada

O notebook recebe `data/colab_bundle.zip` atualizado, instala o ambiente isolado
e monta o Drive. Mostra prévias de uma imagem de treino de cada origem antes
de iniciar os experimentos. Nenhuma imagem do teste é processada ou avaliada.

Treino e validação recebem o mesmo processamento. Todas as variantes, inclusive
RGB e cinza, são gravadas como PNG sem perdas para evitar diferenças de compressão.
Os TXT são copiados byte a byte; não há mudança de caixas nem redistribuição.
O fingerprint inclui o manifesto de treino/validação, os bytes reais das imagens
e rótulos, parâmetros, código de processamento e versões das bibliotecas.

Somente uma variante é gerada por vez no armazenamento temporário do Colab.
Após salvar resultado e ranking, seu cache é removido; os dados originais e
resultados no Drive são preservados. Com `--keep-data`, o cache é mantido.
Na retomada, as imagens são regeneradas deterministamente a partir dos originais.
Pastas incompletas sem checkpoint ou assinaturas incompatíveis não são sobrescritas.
Não executar a grade simultaneamente em duas sessões.

Resultados: `Meu Drive/what-is-the-book-2/runs/preprocessing/grid/<variante>/`.
Cada pasta contém pesos, métricas, gráficos e `result.json`. A configuração
solicitada fica em `<variante>_request.json`; o ranking usa mAP50–95 de validação.
O notebook mostra resumos por época e salva o log integral no Drive. O teste
curto é opcional e vai para uma pasta separada (`smoke`).

Ao escolher uma variante, aplicar a mesma função e parâmetros também nas futuras
imagens de inferência e na avaliação final do teste. Como o cache é temporário,
os YAMLs de cada variante apontam para dados que precisam ser regenerados para
uma nova avaliação. Os arquivos de resultado e pesos permanecem disponíveis.

## Execução

```powershell
uv run python -m src.detection.train_preprocessing --preview
uv run python src/preprocess/package_colab.py
```

Abra o novo notebook no Colab e envie o ZIP atualizado. O treinamento completo
é iniciado explicitamente pela última célula. Se a sessão cair, restaure o
ambiente e execute a mesma célula: resultados concluídos são pulados e checkpoints
incompletos retomados. Após a conclusão, comparar também precisão, recall, mAP50,
exemplos visuais e tempo de processamento, não apenas a melhor mAP50–95.

Os treinamentos não foram executados localmente nesta preparação. A conclusão
sobre qual filtro funciona melhor depende dos resultados no Colab.

Referências: [filtros OpenCV](https://docs.opencv.org/4.13.0/d4/d13/tutorial_py_filtering.html),
[gradientes OpenCV](https://docs.opencv.org/4.13.0/d5/d0f/tutorial_py_gradients.html),
[Canny OpenCV](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).
