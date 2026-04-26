"""
Модуль прогнозирования продаж.

Реализованы три метода интеллектуального анализа данных:
1. Random Forest — ансамбль деревьев решений (бэггинг, sklearn)
2. LightGBM     — градиентный бустинг с расширенным набором признаков
3. XGBoost      — градиентный бустинг (базовая версия для сравнения)

Каждая функция возвращает:
- forecast_df : DataFrame с прогнозными значениями
- metrics     : словарь метрик качества (MAE, RMSE, MAPE)
- fig         : Plotly-фигура с визуализацией прогноза
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    return {
        "MAE":  round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
    }


def _forecast_plot(
    history_ds, history_y,
    forecast_ds, forecast_y,
    test_ds=None, test_y=None,
    title: str = "Прогноз продаж",
    lower=None, upper=None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_ds, y=history_y,
        mode="lines", name="История",
        line=dict(color="#2563EB", width=2),
    ))

    if test_ds is not None and test_y is not None:
        fig.add_trace(go.Scatter(
            x=test_ds, y=test_y,
            mode="lines+markers", name="Факт (тест)",
            line=dict(color="#16A34A", width=2, dash="dot"),
        ))

    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(forecast_ds) + list(forecast_ds)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor="rgba(239,68,68,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Доверит. интервал",
        ))

    fig.add_trace(go.Scatter(
        x=forecast_ds, y=forecast_y,
        mode="lines+markers", name="Прогноз",
        line=dict(color="#EF4444", width=2),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Выручка",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _make_ci(forecast_values: np.ndarray, residual_std: float) -> tuple:
    """Доверительный интервал ±1.96σ на основе ошибки на тесте."""
    lower = forecast_values - 1.96 * residual_std
    upper = forecast_values + 1.96 * residual_std
    return lower, upper


# ─── 1. Random Forest ─────────────────────────────────────────────────────────

def run_random_forest(daily_df: pd.DataFrame, forecast_days: int = 30):
    """
    Random Forest — ансамбль независимых деревьев решений (бэггинг).

    Отличие от XGBoost/LightGBM: деревья строятся параллельно и независимо,
    результат — среднее по всем деревьям. Бустинг исправляет ошибки итерационно,
    Random Forest усредняет разнообразие — разные подходы к снижению дисперсии.

    Признаки: лаги 1/2/3/7/14/21/28/30 дней, скользящие средние и медиана 7/14/30,
    скользящее стандартное отклонение 7/30, день недели, месяц, квартал,
    номер дня года, флаг выходного дня.

    Прогноз строится рекуррентно: каждый следующий день предсказывается
    на основе предыдущих прогнозных значений.

    Args:
        daily_df:      DataFrame ['ds','y']
        forecast_days: горизонт прогноза в днях

    Returns:
        forecast_df, metrics, fig
    """
    from sklearn.ensemble import RandomForestRegressor

    df = daily_df.copy()
    df = df.set_index("ds").asfreq("D").fillna(0).reset_index()

    def make_features(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["dayofweek"]  = d["ds"].dt.dayofweek
        d["month"]      = d["ds"].dt.month
        d["quarter"]    = d["ds"].dt.quarter
        d["dayofyear"]  = d["ds"].dt.dayofyear
        d["year"]       = d["ds"].dt.year
        d["is_weekend"] = (d["ds"].dt.dayofweek >= 5).astype(int)
        for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
            d[f"lag_{lag}"] = d["y"].shift(lag)
        d["rolling_7"]   = d["y"].shift(1).rolling(7).mean()
        d["rolling_14"]  = d["y"].shift(1).rolling(14).mean()
        d["rolling_30"]  = d["y"].shift(1).rolling(30).mean()
        d["median_7"]    = d["y"].shift(1).rolling(7).median()
        d["std_7"]       = d["y"].shift(1).rolling(7).std()
        d["std_30"]      = d["y"].shift(1).rolling(30).std()
        return d

    df = make_features(df)
    df = df.dropna()

    feature_cols = [
        "dayofweek", "month", "quarter", "dayofyear", "year", "is_weekend",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28", "lag_30",
        "rolling_7", "rolling_14", "rolling_30", "median_7", "std_7", "std_30",
    ]

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[feature_cols], train["y"]
    X_test,  y_test  = test[feature_cols],  test["y"]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_pred    = model.predict(X_test)
    metrics      = _compute_metrics(y_test.values, test_pred)
    residual_std = float(np.std(y_test.values - test_pred))

    # Рекуррентный прогноз в будущее
    last_known  = df.copy()
    future_rows = []
    last_date   = df["ds"].max()

    for i in range(1, forecast_days + 1):
        next_date = last_date + pd.Timedelta(days=i)
        row = {
            "ds":         next_date,
            "dayofweek":  next_date.dayofweek,
            "month":      next_date.month,
            "quarter":    next_date.quarter,
            "dayofyear":  next_date.dayofyear,
            "year":       next_date.year,
            "is_weekend": int(next_date.dayofweek >= 5),
        }
        all_y = list(last_known["y"].values) + [r["y_hat"] for r in future_rows]
        for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
            row[f"lag_{lag}"] = all_y[-lag] if len(all_y) >= lag else 0
        row["rolling_7"]  = np.mean(all_y[-7:])   if len(all_y) >= 7  else np.mean(all_y)
        row["rolling_14"] = np.mean(all_y[-14:])  if len(all_y) >= 14 else np.mean(all_y)
        row["rolling_30"] = np.mean(all_y[-30:])  if len(all_y) >= 30 else np.mean(all_y)
        row["median_7"]   = np.median(all_y[-7:]) if len(all_y) >= 7  else np.median(all_y)
        row["std_7"]      = np.std(all_y[-7:])    if len(all_y) >= 7  else np.std(all_y)
        row["std_30"]     = np.std(all_y[-30:])   if len(all_y) >= 30 else np.std(all_y)

        X_future     = pd.DataFrame([row])[feature_cols]
        row["y_hat"] = float(model.predict(X_future)[0])
        future_rows.append(row)

    forecast_df = pd.DataFrame(future_rows)[["ds", "y_hat"]]
    lower, upper = _make_ci(forecast_df["y_hat"].values, residual_std)
    forecast_df["lower"] = lower
    forecast_df["upper"] = upper

    fig = _forecast_plot(
        history_ds=train["ds"], history_y=train["y"],
        forecast_ds=forecast_df["ds"], forecast_y=forecast_df["y_hat"],
        test_ds=test["ds"], test_y=test["y"],
        title="Random Forest — Прогноз ежедневной выручки",
        lower=lower, upper=upper,
    )

    return forecast_df, metrics, fig


# ─── 2. LightGBM ──────────────────────────────────────────────────────────────

def run_lightgbm(daily_df: pd.DataFrame, forecast_days: int = 30):
    """
    LightGBM — градиентный бустинг на деревьях решений (Microsoft).

    Отличия от XGBoost:
    - Растит деревья листьями (leaf-wise), а не уровнями (level-wise)
    - Более гибкая регуляризация (num_leaves, min_child_samples)
    - Расширенный набор признаков: лаги 1/7/14/21/30 дней,
      скользящие средние 7/14/30 дней, скользящее стандартное отклонение,
      флаг выходного дня

    Прогноз строится рекуррентно: каждый следующий день предсказывается
    на основе предыдущих прогнозных значений.

    Args:
        daily_df:      DataFrame ['ds','y']
        forecast_days: горизонт прогноза в днях

    Returns:
        forecast_df, metrics, fig
    """
    from lightgbm import LGBMRegressor

    df = daily_df.copy()
    df = df.set_index("ds").asfreq("D").fillna(0).reset_index()

    def make_features(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["dayofweek"]  = d["ds"].dt.dayofweek
        d["month"]      = d["ds"].dt.month
        d["quarter"]    = d["ds"].dt.quarter
        d["dayofyear"]  = d["ds"].dt.dayofyear
        d["year"]       = d["ds"].dt.year
        d["is_weekend"] = (d["ds"].dt.dayofweek >= 5).astype(int)
        for lag in [1, 7, 14, 21, 30]:
            d[f"lag_{lag}"] = d["y"].shift(lag)
        d["rolling_7"]  = d["y"].shift(1).rolling(7).mean()
        d["rolling_14"] = d["y"].shift(1).rolling(14).mean()
        d["rolling_30"] = d["y"].shift(1).rolling(30).mean()
        d["std_7"]      = d["y"].shift(1).rolling(7).std()
        return d

    df = make_features(df)
    df = df.dropna()

    feature_cols = [
        "dayofweek", "month", "quarter", "dayofyear", "year", "is_weekend",
        "lag_1", "lag_7", "lag_14", "lag_21", "lag_30",
        "rolling_7", "rolling_14", "rolling_30", "std_7",
    ]

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[feature_cols], train["y"]
    X_test,  y_test  = test[feature_cols],  test["y"]

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    metrics   = _compute_metrics(y_test.values, test_pred)

    residual_std = float(np.std(y_test.values - test_pred))

    # Рекуррентный прогноз в будущее
    last_known  = df.copy()
    future_rows = []
    last_date   = df["ds"].max()

    for i in range(1, forecast_days + 1):
        next_date = last_date + pd.Timedelta(days=i)
        row = {
            "ds":         next_date,
            "dayofweek":  next_date.dayofweek,
            "month":      next_date.month,
            "quarter":    next_date.quarter,
            "dayofyear":  next_date.dayofyear,
            "year":       next_date.year,
            "is_weekend": int(next_date.dayofweek >= 5),
        }
        all_y = list(last_known["y"].values) + [r["y_hat"] for r in future_rows]
        row["lag_1"]  = all_y[-1]  if len(all_y) >= 1  else 0
        row["lag_7"]  = all_y[-7]  if len(all_y) >= 7  else 0
        row["lag_14"] = all_y[-14] if len(all_y) >= 14 else 0
        row["lag_21"] = all_y[-21] if len(all_y) >= 21 else 0
        row["lag_30"] = all_y[-30] if len(all_y) >= 30 else 0
        row["rolling_7"]  = np.mean(all_y[-7:])  if len(all_y) >= 7  else np.mean(all_y)
        row["rolling_14"] = np.mean(all_y[-14:]) if len(all_y) >= 14 else np.mean(all_y)
        row["rolling_30"] = np.mean(all_y[-30:]) if len(all_y) >= 30 else np.mean(all_y)
        row["std_7"]      = np.std(all_y[-7:])   if len(all_y) >= 7  else np.std(all_y)

        X_future     = pd.DataFrame([row])[feature_cols]
        row["y_hat"] = float(model.predict(X_future)[0])
        future_rows.append(row)

    forecast_df = pd.DataFrame(future_rows)[["ds", "y_hat"]]
    lower, upper = _make_ci(forecast_df["y_hat"].values, residual_std)
    forecast_df["lower"] = lower
    forecast_df["upper"] = upper

    fig = _forecast_plot(
        history_ds=train["ds"], history_y=train["y"],
        forecast_ds=forecast_df["ds"], forecast_y=forecast_df["y_hat"],
        test_ds=test["ds"], test_y=test["y"],
        title="LightGBM — Прогноз ежедневной выручки",
        lower=lower, upper=upper,
    )

    return forecast_df, metrics, fig


# ─── 3. XGBoost ───────────────────────────────────────────────────────────────

def run_xgboost(daily_df: pd.DataFrame, forecast_days: int = 30):
    """
    XGBoost — градиентный бустинг (eXtreme Gradient Boosting).

    Признаки: лаги 1/7/14/30 дней, скользящие средние 7/30 дней,
    день недели, месяц, квартал, номер дня года.
    Прогноз строится рекуррентно.

    Args:
        daily_df:      DataFrame ['ds','y']
        forecast_days: горизонт прогноза в днях

    Returns:
        forecast_df, metrics, fig
    """
    from xgboost import XGBRegressor

    df = daily_df.copy()
    df = df.set_index("ds").asfreq("D").fillna(0).reset_index()

    def make_features(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["dayofweek"] = d["ds"].dt.dayofweek
        d["month"]     = d["ds"].dt.month
        d["quarter"]   = d["ds"].dt.quarter
        d["dayofyear"] = d["ds"].dt.dayofyear
        d["year"]      = d["ds"].dt.year
        for lag in [1, 7, 14, 30]:
            d[f"lag_{lag}"] = d["y"].shift(lag)
        d["rolling_7"]  = d["y"].shift(1).rolling(7).mean()
        d["rolling_30"] = d["y"].shift(1).rolling(30).mean()
        return d

    df = make_features(df)
    df = df.dropna()

    feature_cols = [
        "dayofweek", "month", "quarter", "dayofyear", "year",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_7", "rolling_30",
    ]

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[feature_cols], train["y"]
    X_test,  y_test  = test[feature_cols],  test["y"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    test_pred    = model.predict(X_test)
    metrics      = _compute_metrics(y_test.values, test_pred)
    residual_std = float(np.std(y_test.values - test_pred))

    last_known  = df.copy()
    future_rows = []
    last_date   = df["ds"].max()

    for i in range(1, forecast_days + 1):
        next_date = last_date + pd.Timedelta(days=i)
        row = {
            "ds":         next_date,
            "dayofweek":  next_date.dayofweek,
            "month":      next_date.month,
            "quarter":    next_date.quarter,
            "dayofyear":  next_date.dayofyear,
            "year":       next_date.year,
        }
        all_y = list(last_known["y"].values) + [r["y_hat"] for r in future_rows]
        row["lag_1"]  = all_y[-1]  if len(all_y) >= 1  else 0
        row["lag_7"]  = all_y[-7]  if len(all_y) >= 7  else 0
        row["lag_14"] = all_y[-14] if len(all_y) >= 14 else 0
        row["lag_30"] = all_y[-30] if len(all_y) >= 30 else 0
        row["rolling_7"]  = np.mean(all_y[-7:])  if len(all_y) >= 7  else np.mean(all_y)
        row["rolling_30"] = np.mean(all_y[-30:]) if len(all_y) >= 30 else np.mean(all_y)

        X_future     = pd.DataFrame([row])[feature_cols]
        row["y_hat"] = float(model.predict(X_future)[0])
        future_rows.append(row)

    forecast_df = pd.DataFrame(future_rows)[["ds", "y_hat"]]
    lower, upper = _make_ci(forecast_df["y_hat"].values, residual_std)
    forecast_df["lower"] = lower
    forecast_df["upper"] = upper

    fig = _forecast_plot(
        history_ds=train["ds"], history_y=train["y"],
        forecast_ds=forecast_df["ds"], forecast_y=forecast_df["y_hat"],
        test_ds=test["ds"], test_y=test["y"],
        title="XGBoost — Прогноз ежедневной выручки",
        lower=lower, upper=upper,
    )

    return forecast_df, metrics, fig


# ─── Сравнение прогнозов на одной оси ────────────────────────────────────────

def compare_forecasts_chart(forecasts: dict, history_ds=None, history_y=None) -> go.Figure:
    """
    Строит все прогнозы на одном графике для визуального сравнения.

    Args:
        forecasts:  {"Random Forest": forecast_df, "LightGBM": ..., "XGBoost": ...}
        history_ds: опционально — исторические даты для контекста (последние 60 дней)
        history_y:  опционально — исторические значения

    Returns:
        Plotly Figure
    """
    palette = {
        "Random Forest": "#2563EB",
        "LightGBM":      "#16A34A",
        "XGBoost":       "#EF4444",
    }
    fig = go.Figure()

    if history_ds is not None and history_y is not None:
        fig.add_trace(go.Scatter(
            x=history_ds, y=history_y,
            mode="lines", name="История (последние 60 дн.)",
            line=dict(color="#94A3B8", width=1.5, dash="dot"),
        ))

    for name, fc_df in forecasts.items():
        color = palette.get(name, "#999999")
        fig.add_trace(go.Scatter(
            x=fc_df["ds"], y=fc_df["y_hat"],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Сравнение прогнозов всех моделей",
        xaxis_title="Дата",
        yaxis_title="Выручка",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ─── Сравнение метрик ─────────────────────────────────────────────────────────

def compare_models(metrics_dict: dict) -> go.Figure:
    """
    Строит сравнительный столбчатый график метрик всех трёх моделей.

    Args:
        metrics_dict: {"Random Forest": {...}, "LightGBM": {...}, "XGBoost": {...}}

    Returns:
        Plotly Figure
    """
    models  = list(metrics_dict.keys())
    metrics = ["MAE", "RMSE", "MAPE"]
    colors  = ["#2563EB", "#16A34A", "#EF4444"]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        values = [metrics_dict[m].get(metric, 0) for m in models]
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=values,
            marker_color=color,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        title="Сравнение метрик качества моделей",
        barmode="group",
        xaxis_title="Модель",
        yaxis_title="Значение метрики",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
