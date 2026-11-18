# Caminhos portáveis

Por padrão, aplicação, scripts PowerShell e notebook de avaliação usam a raiz
do repositório para `.venv`, `models`, `runs` e caches. Os dados de origem ficam
em `data/unified` dentro do repositório, independentemente do armazenamento.
O notebook encontra a raiz a partir do diretório atual ou de seus pais.

Para guardar ambiente, modelos e resultados em outro disco, copie
`settings.example.json` para `settings.local.json` e altere `storage`:

```json
{
  "storage": "../book-storage"
}
```

Também aceita um caminho absoluto. Caminhos relativos são resolvidos em relação
à raiz do repositório, nunca ao diretório de onde o comando foi chamado.
O arquivo local é ignorado pelo Git. Alterar essa configuração não move arquivos:
aponte para a pasta que já contém `.venv`, `models` e `runs`, ou prepare um novo
ambiente nela. Neste PC a configuração local preserva o armazenamento existente.

Precedência: argumento `-Storage`/`--storage`, variável de ambiente `BOOK_STORAGE`,
`settings.local.json`, raiz do repositório. Exemplo PowerShell:

```powershell
$env:BOOK_STORAGE = '../book-storage'
.\start_app.ps1
```

No Colab, os caminhos `/content` e `/content/drive/MyDrive` são específicos do
ambiente e permanecem na célula de configuração, sem nomes de usuários locais.
Para executar o notebook no PC, selecione um kernel Python com `ipykernel`
disponível; ele inicia a avaliação no Python do armazenamento configurado.
