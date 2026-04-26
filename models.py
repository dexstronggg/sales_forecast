"""
Модуль прогнозирования продаж.

Реализованы три метода интеллектуального анализа данных:
1. ARIMA  — классический статистический метод временных рядов
2. Prophet — метод от Meta для временных рядов с сезонностью
3. XGBoost — градиентный бустинг (машинное обучение)

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
    """
    Рассчитывает метрики качества прогноза.

    MAE  — средняя абсолютная ошибка (в единицах выручки)
    RMSE — корень из среднеквадратической ошибки (штрафует крупные ошибки)
    MAPE — средняя абсолютная процентная ошибка (в %)
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Защита от деления на ноль
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
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
    """
    Строит единый Plotly-график:
    - синяя линия  : исторические данные
    - зелёная линия: тестовые (реальные) значения
    - красная линия: прогноз модели
    - серая полоса : доверительный интервал (если передан)
    """
    fig = go.Figure()

    # Исторические данные
    fig.add_trace(go.Scatter(
        x=history_ds, y=history_y,
        mode="lines", name="История",
        line=dict(color="#2563EB", width=2),
    ))

    # Реальные значения тестовой выборки
    if test_ds is not None and test_y is not None:
        fig.add_trace(go.Scatter(
            x=test_ds, y=test_y,
            mode="lines+markers", name="Факт (тест)",
            line=dict(color="#16A34A", width=2, dash="dot"),
        ))

    # Доверительный интервал
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(forecast_ds) + list(forecast_ds)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor="rgba(239,68,68,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Доверит. интервал",
        ))

    # Прогноз
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


# ─── 1. ARIMA ─────────────────────────────────────────────────────────────────

def run_arima(daily_df: pd.DataFrame, forecast_days: int = 30):
    """
    Метод ARIMA (AutoRegressive Integrated Moving Average).

    Принцип работы:
    - AR(p): авторегрессия — текущее значение зависит от p предыдущих
    - I(d) : дифференцирование — убирает нестационарность ряда
    - MA(q): скользящее среднее — учёт ошибок прошлых прогнозов

    Параметры (p,d,q) = (5,1,0) — подобраны эмпирически для данного ряда.

    Args:
        daily_df:      DataFrame с колонками ['ds','y'] (ежедневные данные)
        forecast_days: горизонт прогноза в днях

    Returns:
        forecast_df, metrics, fig
    """
    from statsmodels.tsa.arima.model import ARIMA

    series = daily_df.set_index("ds")["y"].asfreq("D").fillna(0)

    # Разбивка: 80% обучение, 20% тест
    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]

    # Обучение модели
    model = ARIMA(train, order=(5, 1, 0))
    fitted = model.fit()

    # Прогноз на тестовый период
    test_pred = fitted.forecast(steps=len(test))

    # Прогноз в будущее
    future_model = ARIMA(series, order=(5, 1, 0)).fit()
    future_fc    = future_model.get_forecast(steps=forecast_days)
    future_mean  = future_fc.predicted_mean
    future_ci    = future_fc.conf_int()

    # Будущие даты
    last_date    = series.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=forecast_days)

    forecast_df = pd.DataFrame({
        "ds":    future_dates,
        "y_hat": future_mean.values,
        "lower": future_ci.iloc[:, 0].values,
        "upper": future_ci.iloc[:, 1].values,
    })

    metrics = _compute_metrics(test.values, test_pred.values)

    fig = _forecast_plot(
        history_ds=train.index, history_y=train.values,
        forecast_ds=future_dates, forecast_y=future_mean.values,
        test_ds=test.index, test_y=test.values,
        title="ARIMA — Прогноз ежедневной выручки",
        lower=future_ci.iloc[:, 0].values,
        upper=future_ci.iloc[:, 1].values,
    )

    return forecast_df, metrics, fig


# ─── 2. Prophet ───────────────────────────────────────────────────────────────

def run_prophet(daily_df: pd.DataFrame, forecast_days: int = 30):
    """
    Метод Facebook Prophet.

    Принцип работы:
    - Декомпозиция ряда: тренд + годовая сезонность + недельная сезонность
    - Тренд моделируется кусочно-линейной функцией с точками излома
    - Автоматически учитывает праздники и аномалии

    Преимущество перед ARIMA: не требует стационарности,
    легко масштабируется на длинные горизонты.

    Args:
        daily_df:      DataFrame ['ds','y']
        forecast_days: горизонт прогноза в днях

    Returns:
        forecast_df, metrics, fig
    """
    from prophet import Prophet

    # Разбивка 80/20
    split = int(len(daily_df) * 0.8)
    train = daily_df.iloc[:split].copy()
    test  = daily_df.iloc[split:].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    model.fit(train)

    # Прогноз на тест
    test_future = model.make_future_dataframe(periods=len(test), include_history=False)
    test_pred   = model.predict(test_future)

    # Прогноз в будущее от конца всего ряда
    all_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    all_model.fit(daily_df)
    future      = all_model.make_future_dataframe(periods=forecast_days)
    forecast    = all_model.predict(future)
    future_only = forecast.tail(forecast_days)

    forecast_df = future_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={"yhat": "y_hat", "yhat_lower": "lower", "yhat_upper": "upper"}
    )

    metrics = _compute_metrics(
        test["y"].values,
        test_pred["yhat"].values,
    )

    fig = _forecast_plot(
        history_ds=train["ds"], history_y=train["y"],
        forecast_ds=future_only["ds"], forecast_y=future_only["yhat"],
        test_ds=test["ds"], test_y=test["y"],
        title="Prophet — Прогноз ежедневной выручки",
        lower=future_only["yhat_lower"].values,
        upper=future_only["yhat_upper"].values,
    )

    return forecast_df, metrics, fig


# ─── 3. XGBoost ───────────────────────────────────────────────────────────────

def run_xgboost(daily_df: pd.DataFrame, forecast_days: int = 30):
    """
    Метод XGBoost (eXtreme Gradient Boosting).

    Принцип работы:
    - Строит ансамбль решающих деревьев последовательно
    - Каждое новое дерево исправляет ошибки предыдущего
    - Признаки: lag-значения (продажи 1, 7, 14, 30 дней назад),
      день недели, месяц, квартал, номер дня года

    Преимущество: умеет использовать внешние признаки (цена, категория)
    и хорошо улавливает нелинейные зависимости.

    Args:
        daily_df:      DataFrame ['ds','y']
        forecast_days: горизонт прогноза в днях

    Returns:
        forecast_df, metrics, fig
    """
    from xgboost import XGBRegressor

    df = daily_df.copy()
    df = df.set_index("ds").asfreq("D").fillna(0).reset_index()

    # ── Создание временных признаков ──
    def make_features(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["dayofweek"] = d["ds"].dt.dayofweek
        d["month"]     = d["ds"].dt.month
        d["quarter"]   = d["ds"].dt.quarter
        d["dayofyear"] = d["ds"].dt.dayofyear
        d["year"]      = d["ds"].dt.year
        # Лаговые признаки: продажи N дней назад
        for lag in [1, 7, 14, 30]:
            d[f"lag_{lag}"] = d["y"].shift(lag)
        # Скользящие средние
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

    # Разбивка 80/20
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

    test_pred = model.predict(X_test)
    metrics   = _compute_metrics(y_test.values, test_pred)

    # ── Рекуррентный прогноз в будущее ──
    last_known = df.copy()
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
        # Лаговые признаки из уже известных + предсказанных значений
        all_y = list(last_known["y"].values) + [r["y_hat"] for r in future_rows]
        row["lag_1"]  = all_y[-1]  if len(all_y) >= 1  else 0
        row["lag_7"]  = all_y[-7]  if len(all_y) >= 7  else 0
        row["lag_14"] = all_y[-14] if len(all_y) >= 14 else 0
        row["lag_30"] = all_y[-30] if len(all_y) >= 30 else 0
        row["rolling_7"]  = np.mean(all_y[-7:])  if len(all_y) >= 7  else np.mean(all_y)
        row["rolling_30"] = np.mean(all_y[-30:]) if len(all_y) >= 30 else np.mean(all_y)

        X_future    = pd.DataFrame([row])[feature_cols]
        row["y_hat"] = float(model.predict(X_future)[0])
        future_rows.append(row)

    forecast_df = pd.DataFrame(future_rows)[["ds", "y_hat"]]
    forecast_df["lower"] = forecast_df["y_hat"] * 0.85
    forecast_df["upper"] = forecast_df["y_hat"] * 1.15

    fig = _forecast_plot(
        history_ds=train["ds"], history_y=train["y"],
        forecast_ds=forecast_df["ds"], forecast_y=forecast_df["y_hat"],
        test_ds=test["ds"], test_y=test["y"],
        title="XGBoost — Прогноз ежедневной выручки",
        lower=forecast_df["lower"].values,
        upper=forecast_df["upper"].values,
    )

    return forecast_df, metrics, fig


# ─── Сравнение моделей ────────────────────────────────────────────────────────

def compare_models(metrics_dict: dict) -> go.Figure:
    """
    Строит сравнительный столбчатый график метрик всех трёх моделей.

    Args:
        metrics_dict: {"ARIMA": {...}, "Prophet": {...}, "XGBoost": {...}}

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
