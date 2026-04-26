"""
Модуль работы с базой данных SQLite.

Структура БД:
- Таблица forecasts: хранит результаты каждого прогноза
- Таблица forecast_values: хранит прогнозные значения по датам

Используется для:
- Сохранения результатов прогнозирования
- Просмотра истории прогнозов
- Сравнения прогнозов между собой
"""

import sqlite3
import pandas as pd
from datetime import datetime


DB_PATH = "sales_forecast.db"


def init_db(db_path: str = DB_PATH) -> None:
    """
    Инициализирует базу данных — создаёт таблицы если их нет.
    Вызывается один раз при запуске приложения.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Таблица заголовков прогнозов
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name  TEXT    NOT NULL,
            created_at  TEXT    NOT NULL,
            horizon     INTEGER NOT NULL,
            mae         REAL,
            rmse        REAL,
            mape        REAL
        )
    """)

    # Таблица значений прогноза
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_values (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_id INTEGER NOT NULL,
            ds          TEXT    NOT NULL,
            y_hat       REAL    NOT NULL,
            lower       REAL,
            upper       REAL,
            FOREIGN KEY (forecast_id) REFERENCES forecasts(id)
        )
    """)

    conn.commit()
    conn.close()


def save_forecast(
    model_name: str,
    metrics: dict,
    forecast_df: pd.DataFrame,
    horizon: int,
    db_path: str = DB_PATH,
) -> int:
    """
    Сохраняет результат прогноза в базу данных.

    Args:
        model_name:  название модели ("Random Forest", "LightGBM", "XGBoost")
        metrics:     словарь {"MAE": ..., "RMSE": ..., "MAPE": ...}
        forecast_df: DataFrame с колонками ['ds','y_hat','lower','upper']
        horizon:     горизонт прогноза в днях
        db_path:     путь к файлу БД

    Returns:
        id сохранённого прогноза
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Сохраняем заголовок
    cur.execute("""
        INSERT INTO forecasts (model_name, created_at, horizon, mae, rmse, mape)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        model_name,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        horizon,
        metrics.get("MAE"),
        metrics.get("RMSE"),
        metrics.get("MAPE"),
    ))
    forecast_id = cur.lastrowid

    # Сохраняем значения
    for _, row in forecast_df.iterrows():
        cur.execute("""
            INSERT INTO forecast_values (forecast_id, ds, y_hat, lower, upper)
            VALUES (?, ?, ?, ?, ?)
        """, (
            forecast_id,
            str(row["ds"])[:10],
            float(row["y_hat"]),
            float(row.get("lower", row["y_hat"])),
            float(row.get("upper", row["y_hat"])),
        ))

    conn.commit()
    conn.close()
    return forecast_id


def load_forecasts(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Загружает список всех сохранённых прогнозов.

    Returns:
        DataFrame с историей прогнозов
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT id, model_name, created_at, horizon, mae, rmse, mape
        FROM forecasts
        ORDER BY created_at DESC
    """, conn)
    conn.close()
    return df


def load_forecast_values(forecast_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Загружает прогнозные значения для конкретного прогноза.

    Args:
        forecast_id: ID прогноза из таблицы forecasts

    Returns:
        DataFrame с колонками ['ds','y_hat','lower','upper']
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT ds, y_hat, lower, upper
        FROM forecast_values
        WHERE forecast_id = ?
        ORDER BY ds
    """, conn, params=(forecast_id,))
    conn.close()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def delete_forecast(forecast_id: int, db_path: str = DB_PATH) -> None:
    """
    Удаляет прогноз и все его значения из базы данных.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("DELETE FROM forecast_values WHERE forecast_id = ?", (forecast_id,))
    cur.execute("DELETE FROM forecasts WHERE id = ?", (forecast_id,))
    conn.commit()
    conn.close()
