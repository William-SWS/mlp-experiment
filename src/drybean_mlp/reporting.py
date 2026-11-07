"""Gravação de tabelas e figuras nos diretórios lidos pelo relatório LaTeX."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from drybean_mlp.data import PROJECT_ROOT

CSV_DIR = PROJECT_ROOT / "results" / "csv"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "report" / "tables"


def save_table(
    table: pd.DataFrame,
    name: str,
    digits: int = 4,
    column_digits: Mapping[str, int] | None = None,
    csv_dir: Path = CSV_DIR,
    table_dir: Path = TABLE_DIR,
) -> pd.DataFrame:
    """Grava `<name>.csv` e o fragmento `<name>.tex` usado com `\\input` no relatório.

    `column_digits` define casas decimais por coluna no LaTeX; o CSV mantém os valores completos.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_dir / f"{name}.csv", index=False)
    # Com `decimal`, o pandas espera texto do formatador de toda coluna, inclusive as inteiras.
    integers = table.select_dtypes(include=["integer", "bool"]).columns
    printable = table.astype(dict.fromkeys(integers, str))
    for column, places in (column_digits or {}).items():
        if column in table and column not in integers:
            printable[column] = table[column].map(
                lambda value, places=places: "--"
                if pd.isna(value)
                else f"{value:.{places}f}".replace(".", ",")
            )
    latex = printable.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        float_format=f"{{:.{digits}f}}".format,
        decimal=",",
    )
    (table_dir / f"{name}.tex").write_text(latex, encoding="utf-8")
    return table


def save_figure(figure: plt.Figure, name: str, figure_dir: Path = FIGURE_DIR) -> plt.Figure:
    """Grava `<name>.pdf` (vetorial, para o relatório) e `<name>.png`; devolve a figura."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(figure_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    return figure
