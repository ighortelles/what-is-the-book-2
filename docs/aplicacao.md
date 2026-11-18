# Aplicação local de detecção

Inicie na raiz do projeto:

```powershell
.\start_app.ps1
```

Abra `http://127.0.0.1:8765` no navegador. Envie uma imagem e clique em
**Comparar detecções**. A aplicação executa o melhor modelo sem filtros e o melhor
com preprocessamento, mostrando as imagens lado a lado (empilhadas no celular).
Ambos usam o mesmo limiar de confiança, preservando a resolução de treinamento
de cada modelo. Há contagens e downloads independentes. A contagem corresponde
às detecções; não identifica títulos e pode haver falsos positivos ou livros omitidos.

O script usa o ambiente já instalado em `.venv`, sem instalar
dependências adicionais. Inferência em CPU, para preservar a GPU do treinamento.
A aplicação pode consumir CPU e memória ao mesmo tempo que outros processos.
Para mudar a porta ou o armazenamento, use `-Port` ou `-Storage`.

## Modelo do Drive

Para o exp_05, baixe:

- `what-is-the-book-2/runs/grid/exp_05/weights/best.pt`
- `what-is-the-book-2/runs/grid/exp_05/result.json`

Coloque ambos em `models\exp_05\` e clique em
**Atualizar modelos do ranking**. A seleção usa o maior mAP50–95 registrado em
`runs/final_test_all/ranking.json`, entre os modelos disponíveis de cada grupo.
O controle `original` da grade de preprocessamento pertence ao grupo sem filtros.
Na ausência de ranking, usa as métricas de validação e indica isso na interface.
Se faltar um dos grupos, informa o erro e não executa uma comparação parcial.
No ranking atual, os selecionados são exp_05 e bilateral_5.
Outros modelos podem ser colocados em `models/<nome>/`, com os mesmos dois arquivos.
O JSON informa a resolução de entrada e, quando aplicável, a variante de filtro.
Use apenas checkpoints confiáveis, produzidos pelo projeto.

Modelos concluídos em `runs/preprocessing_local/grid` e `runs/grid` também
participam da seleção.
Somente pastas com resultado e checkpoint são carregadas. Cada variante recebe
o mesmo filtro usado no treinamento; as caixas são desenhadas na foto original.
Modelos da grade original usam a imagem sem filtros. O exp_05 usa resolução 960.

## Fotos e processamento

Fotos são processadas em memória e não são salvas pelo servidor. Não são enviadas
ao Drive ou a outro serviço. O servidor escuta apenas em `127.0.0.1`, para uso local.
Não há envio de pesos pela interface. As fotos podem ter até 20 MB e 25 megapixels.
A orientação EXIF é aplicada antes da inferência. O modelo aplica seu
redimensionamento/letterbox padrão; NMS usa IoU 0,7 e até 1000 detecções.
O limiar da interface é para exploração de fotos, não altera avaliações finais.

Arquivos da aplicação: `src/app/server.py`, `src/app/index.html` e `start_app.ps1`.
A execução em segundo plano iniciada pelo assistente registra logs em
`app.log` e `app_errors.log`. Para executar manualmente,
encerre antes a instância existente; Ctrl+C encerra uma execução no terminal.

Os caminhos locais são relativos ao armazenamento configurado, cujo padrão é
a raiz do repositório. Consulte [configuração de caminhos](caminhos.md)
para usar outro disco sem alterar o código.
