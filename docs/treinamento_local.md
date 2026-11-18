# Treinamento local na GTX 1060 de 3 GB

Execute na raiz do repositório, com PowerShell:

```powershell
.\train_local.ps1 -Setup -Smoke
.\train_local.ps1
```

O primeiro comando instala o ambiente no armazenamento configurado e verifica um cálculo e seu
gradiente na GPU antes de executar uma época curta. Requer internet para baixar
dependências e os pesos iniciais YOLOv8n. Não baixa resultados antigos do Drive.
Os dados originais continuam em `data/unified` no repositório.

O segundo comando executa as 11 variantes de filtros e bordas. Repetir esse
comando retoma checkpoints compatíveis e pula resultados concluídos. Não rode
duas instâncias simultaneamente sobre a mesma pasta de resultados.

`configs/yolov8_preprocessing_local.json` mantém resolução 640, até 50 épocas,
AdamW e taxa inicial 0,001, mas usa batch 2, workers 0 e AMP desabilitado para
o perfil local. Todas as variantes locais usam esses mesmos parâmetros; a
variante original é o controle. A mudança de batch impede atribuir diferenças
em relação ao Colab exclusivamente aos filtros. As definições do Colab não foram
alteradas. O conjunto de teste permanece reservado, sem uso nesta grade.

Ambiente e cache de pacotes: `.venv` e `uv-cache`.
Imagens temporárias: `preprocessed`, removidas ao concluir
cada variante. Resultados, checkpoints e ranking:
`runs\preprocessing_local\grid`.
O smoke test usa uma pasta separada `smoke`.

PyTorch 2.7.1 e torchvision 0.22.1 usam wheels CUDA 12.6 explicitamente instalados
após a sincronização do lockfile. O script chama o Python diretamente para
preservar esses wheels. Não use `uv sync` nesse ambiente durante o treinamento;
se precisar reconstruí-lo, use `-Setup` novamente. `-Storage` permite escolher
outro diretório para ambiente, cache e resultados.

## Fila e acompanhamento

`run_local_queue.ps1` executa primeiro o smoke test e somente então a grade,
interrompendo em caso de erro. Usa um bloqueio de arquivo para impedir duas
filas simultâneas e registra a etapa em `training_status.json` no armazenamento.
O parâmetro interno `-WaitForInstallProcess` permite aguardar a instalação inicial.

Na execução iniciada em segundo plano, os logs ficam em
`training.log` e `training_errors.log`. Para acompanhar:

```powershell
Get-Content training_status.json
Get-Content training.log -Tail 25 -Wait
```

Interromper apenas esse comando de leitura com Ctrl+C não interrompe o treinamento.
Mantenha o computador ligado e sem suspensão durante a execução. Não é necessário
manter um notebook ou navegador aberto.

Os caminhos locais são relativos ao armazenamento configurado, cujo padrão é
a raiz do repositório. Consulte [configuração de caminhos](caminhos.md)
para usar outro disco sem alterar o código.
