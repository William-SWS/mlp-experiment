# MLP com backpropagation na base Dry Bean

Experimentos da **Avaliação RNA 2026.2**: uma rede MLP classificando sete variedades de feijão a partir de
16 atributos geométricos ([Dry Bean, UCI](https://doi.org/10.24432/C50S4B)). O trabalho segue as sete etapas do
enunciado, de pesos iniciais até validação cruzada.

| Entrega | Onde |
|---|---|
| Experimento principal | [`notebooks/mlp_dry_bean.ipynb`](notebooks/mlp_dry_bean.ipynb) |
| Relatório | [`report/relatorio.pdf`](report/relatorio.pdf) |
| Código | [`src/drybean_mlp/`](src/drybean_mlp/) |
| Tabelas e figuras | [`results/`](results/) e [`report/tables/`](report/tables/) |

## Resultado

Rede 80×80 com ReLU e softmax, treinada por SGD (LR = 0,01, momento = 0,7), usando 10 atributos de forma.
Em validação cruzada com 10 partições, a Medida-F macro é **0,9436 ± 0,0085**. A regressão logística, nas mesmas
partições, fica em 0,9365.

## Como executar

Requer [uv](https://docs.astral.sh/uv/). A base (cerca de 3 MB) é baixada do UCI no primeiro uso, para `data/`.

```bash
uv sync
```

```bash
uv run jupyter lab notebooks/mlp_dry_bean.ipynb
```

Para executar o notebook inteiro pelo terminal:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/mlp_dry_bean.ipynb
```

Os treinos ficam em cache em `results/.cache`, então só a primeira execução é demorada. Apague essa pasta ao mudar o
código de `training.py`: o cache só percebe alterações na própria função em cache.

Para compilar o relatório, que precisa de `latexmk` e de uma distribuição LaTeX:

```bash
make -C report
```

Para rodar os testes e o lint:

```bash
uv run pytest
```

```bash
uv run ruff check src tests
```

## Código

| Módulo | Responsabilidade |
|---|---|
| `data.py` | Download, remoção de duplicatas, divisão estratificada, subamostragem e ruído de rótulo |
| `training.py` | `train_mlp`: treino com `partial_fit`, z-score no treino, métricas por época e detecção de divergência |
| `experiments.py` | Execução paralela com cache, k-fold, agregação, regras de seleção, ranking de atributos e cronometragem |
| `plots.py` | Gráficos dos experimentos |
| `reporting.py` | Gravação de tabelas (CSV e LaTeX) e figuras |

## Decisões de método

- **`partial_fit` em vez de `warm_start`:** `fit` recria o otimizador a cada chamada e zera a velocidade do momento.
  Com 200 épocas, a perda fica de 3% a 25% maior (a medição está no notebook, na Questão 2).
- **`RandomState` em vez de seed inteira:** uma seed inteira repete o embaralhamento em todas as épocas.
- **Duplicatas removidas antes da divisão:** 68 linhas que poderiam cair ao mesmo tempo no treino e no teste.
- **Tempos medidos à parte:** um treino por vez, em uma thread, com o mínimo de cinco repetições. Os tempos dos treinos
  em paralelo variam com a carga da máquina e não permitem comparar configurações.
