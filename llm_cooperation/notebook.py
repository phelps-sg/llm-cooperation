from typing import Callable

import pandas as pd
import plotly
from IPython.core.display import HTML
from IPython.core.display_functions import display


def graph(fn: Callable, name: str) -> None:
    fig = fn()
    html_filename = f"{name}.html"
    pdf_filename = f"{name}.pdf"
    plotly.offline.plot(fig, filename=html_filename, auto_open=False)
    display(HTML(filename=html_filename))
    fig.write_image(pdf_filename)


def save_table(df: pd.DataFrame, name: str, caption: str) -> pd.DataFrame:
    renderer = df.style.format(decimal=".", thousands=",", precision=2)
    renderer.to_latex(f"../latex/{name}.tex", caption=caption)
    return df
