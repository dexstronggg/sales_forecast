"""
Модуль разведочного анализа данных (EDA).
Все графики строятся с помощью Plotly — интерактивные,
отображаются в Streamlit через st.plotly_chart().
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_revenue_over_time(df: pd.DataFrame) -> go.Figure:
    """
    Линейный график ежемесячной выручки за весь период.
    Показывает общий тренд и сезонность продаж.
    """
    monthly = (
        df.groupby("month_label")["revenue"]
        .sum()
        .reset_index()
        .sort_values("month_label")
    )
    fig = px.line(
        monthly,
        x="month_label",
        y="revenue",
        title="Динамика выручки по месяцам",
        labels={"month_label": "Месяц", "revenue": "Выручка"},
        markers=True,
        color_discrete_sequence=["#2563EB"],
    )
    fig.update_layout(xaxis_tickangle=-45, hovermode="x unified")
    return fig


def plot_category_revenue(df: pd.DataFrame) -> go.Figure:
    """
    Горизонтальная столбчатая диаграмма выручки по категориям товаров.
    """
    cat = (
        df.groupby("category")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue")
    )
    fig = px.bar(
        cat,
        x="revenue",
        y="category",
        orientation="h",
        title="Выручка по категориям товаров",
        labels={"revenue": "Выручка", "category": "Категория"},
        color="revenue",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_mall_revenue(df: pd.DataFrame) -> go.Figure:
    """
    Столбчатая диаграмма выручки по торговым центрам.
    """
    mall = (
        df.groupby("shopping_mall")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    fig = px.bar(
        mall,
        x="shopping_mall",
        y="revenue",
        title="Выручка по торговым центрам",
        labels={"shopping_mall": "Торговый центр", "revenue": "Выручка"},
        color="revenue",
        color_continuous_scale="Teal",
    )
    fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
    return fig


def plot_payment_pie(df: pd.DataFrame) -> go.Figure:
    """
    Круговая диаграмма распределения способов оплаты.
    """
    pay = df["payment_method"].value_counts().reset_index()
    pay.columns = ["payment_method", "count"]
    fig = px.pie(
        pay,
        names="payment_method",
        values="count",
        title="Распределение способов оплаты",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_revenue_by_weekday(df: pd.DataFrame) -> go.Figure:
    """
    Столбчатая диаграмма средней выручки по дням недели.
    Позволяет выявить, в какие дни продажи выше.
    """
    day_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    wd = (
        df.groupby("day_of_week")["revenue"]
        .mean()
        .reset_index()
    )
    wd["day_name"] = wd["day_of_week"].apply(lambda x: day_names[x])
    fig = px.bar(
        wd,
        x="day_name",
        y="revenue",
        title="Средняя выручка по дням недели",
        labels={"day_name": "День недели", "revenue": "Средняя выручка"},
        color="revenue",
        color_continuous_scale="Purples",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_heatmap_month_year(df: pd.DataFrame) -> go.Figure:
    """
    Тепловая карта выручки: год × месяц.
    Наглядно показывает сезонность и рост/спад по годам.
    """
    pivot = (
        df.groupby(["year", "month"])["revenue"]
        .sum()
        .unstack(level=0)
    )
    month_names = [
        "Янв", "Фев", "Мар", "Апр", "Май", "Июн",
        "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"
    ]
    pivot.index = [month_names[i - 1] for i in pivot.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index,
            colorscale="Blues",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Тепловая карта выручки (год × месяц)",
        xaxis_title="Год",
        yaxis_title="Месяц",
    )
    return fig


def plot_top_products(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Топ-N товаров по суммарной выручке.
    """
    top = (
        df.groupby("product_name")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(top_n)
    )
    fig = px.bar(
        top,
        x="revenue",
        y="product_name",
        orientation="h",
        title=f"Топ-{top_n} товаров по выручке",
        labels={"revenue": "Выручка", "product_name": "Товар"},
        color="revenue",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    return fig
