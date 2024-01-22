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


def _repeated_to_long_row(row: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(row).transpose()
    choices = df.Choices.values[0]
    n = len(choices)
    user_choices = [c.user.as_int for c in choices]
    ai_choices = [c.ai.as_int for c in choices]
    df_long = pd.concat([df] * n, axis=0)
    df_long["User_choice"] = user_choices
    df_long["AI_choice"] = ai_choices
    df_long["Round"] = range(n)
    return df_long


def repeated_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            _repeated_to_long_row(df.iloc[i])
            for i in range(len(df))
            if df.iloc[i]["Choices"] is not None
        ]
    )
