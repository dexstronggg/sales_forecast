"""
Модуль загрузки и предобработки данных.
Датасет: Customer Shopping Dataset (Kaggle)
Источник: kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset
Период: 2021–2023, 10 торговых центров Стамбула
"""

import pandas as pd
import numpy as np


# Словарь для добавления названий товаров по категориям
PRODUCT_NAMES = {
    "Clothing": [
        "Мужская куртка", "Женское платье", "Джинсы", "Футболка",
        "Спортивный костюм", "Пальто", "Рубашка", "Блузка"
    ],
    "Shoes": [
        "Кроссовки", "Туфли классические", "Сапоги зимние",
        "Сандалии", "Ботинки", "Слипоны"
    ],
    "Books": [
        "Роман", "Учебник", "Бизнес-книга",
        "Детская книга", "Энциклопедия"
    ],
    "Cosmetics": [
        "Тушь для ресниц", "Тональный крем", "Помада",
        "Духи", "Крем для лица", "Шампунь"
    ],
    "Food & Beverage": [
        "Кофе в зернах", "Шоколад премиум", "Чай",
        "Орехи ассорти", "Оливковое масло"
    ],
    "Toys": [
        "Конструктор LEGO", "Кукла", "Настольная игра",
        "Мягкая игрушка", "Радиоуправляемая машина"
    ],
    "Technology": [
        "Смартфон", "Наушники беспроводные", "Планшет",
        "Умные часы", "Портативная колонка"
    ],
    "Souvenir": [
        "Магнит", "Брелок", "Декоративная тарелка",
        "Статуэтка", "Открытка"
    ],
}


def load_data(filepath: str) -> pd.DataFrame:
    """
    Загружает CSV-файл датасета и выполняет базовую предобработку.

    Шаги предобработки:
    1. Парсинг дат
    2. Добавление столбца product_name по категории
    3. Добавление столбца revenue = price * quantity
    4. Извлечение year, month, day, day_of_week
    5. Удаление дубликатов и строк с пропусками

    Args:
        filepath: путь к CSV-файлу

    Returns:
        Очищенный DataFrame
    """
    df = pd.read_csv(filepath)

    # --- 1. Переименование столбцов для удобства ---
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # --- 2. Парсинг даты ---
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], dayfirst=True)

    # --- 3. Добавление названий товаров ---
    np.random.seed(42)
    df["product_name"] = df["category"].apply(
        lambda cat: np.random.choice(PRODUCT_NAMES.get(cat, ["Прочий товар"]))
    )

    # --- 4. Выручка по строке ---
    df["revenue"] = df["price"] * df["quantity"]

    # --- 5. Временные признаки ---
    df["year"]         = df["invoice_date"].dt.year
    df["month"]        = df["invoice_date"].dt.month
    df["day"]          = df["invoice_date"].dt.day
    df["day_of_week"]  = df["invoice_date"].dt.dayofweek  # 0=пн … 6=вс
    df["month_label"]  = df["invoice_date"].dt.to_period("M").astype(str)

    # --- 6. Удаление пропусков и дубликатов ---
    df = df.dropna()
    df = df.drop_duplicates()

    return df


def get_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные по дням: суммарная выручка за каждый день.
    Используется для обучения моделей прогнозирования (Random Forest, LightGBM, XGBoost).

    Returns:
        DataFrame с колонками ['ds', 'y'] — дата и выручка.
    """
    daily = (
        df.groupby("invoice_date")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"invoice_date": "ds", "revenue": "y"})
        .sort_values("ds")
    )
    return daily


def get_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные по месяцам.

    Returns:
        DataFrame с колонками ['ds', 'y'].
    """
    monthly = (
        df.groupby("month_label")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"month_label": "ds", "revenue": "y"})
        .sort_values("ds")
    )
    monthly["ds"] = pd.to_datetime(monthly["ds"])
    return monthly


def get_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует суммарную выручку по категориям товаров.

    Returns:
        DataFrame с колонками ['category', 'revenue'].
    """
    return (
        df.groupby("category")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )


def get_daily_sales_filtered(
    df: pd.DataFrame,
    category: str = "Все категории",
    mall: str = "Все ТЦ",
) -> pd.DataFrame:
    """
    Агрегирует ежедневную выручку с фильтрацией по категории и/или торговому центру.

    Returns:
        DataFrame ['ds', 'y']
    """
    filtered = df.copy()
    if category != "Все категории":
        filtered = filtered[filtered["category"] == category]
    if mall != "Все ТЦ":
        filtered = filtered[filtered["shopping_mall"] == mall]
    return get_daily_sales(filtered)


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Возвращает словарь с базовой статистикой по датасету.
    Используется для отображения на главной странице дашборда.
    """
    return {
        "total_revenue":    round(df["revenue"].sum(), 2),
        "total_orders":     len(df),
        "avg_order_value":  round(df["revenue"].mean(), 2),
        "num_categories":   df["category"].nunique(),
        "num_malls":        df["shopping_mall"].nunique(),
        "date_range":       f"{df['invoice_date'].min().date()} — {df['invoice_date'].max().date()}",
    }