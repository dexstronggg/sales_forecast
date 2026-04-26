"""
Главный файл Streamlit-приложения.
Информационная система для прогнозирования продаж.

Запуск: streamlit run app.py

Навигация (боковое меню):
  1. Главная         — сводная статистика, загрузка данных
  2. Анализ данных   — EDA-графики
  3. Прогнозирование — выбор модели, настройка, запуск
  4. История         — просмотр сохранённых прогнозов
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import load_data, get_daily_sales, get_monthly_sales, get_summary_stats
from eda import (
    plot_revenue_over_time,
    plot_category_revenue,
    plot_mall_revenue,
    plot_payment_pie,
    plot_revenue_by_weekday,
    plot_heatmap_month_year,
    plot_top_products,
)
from models import run_arima, run_prophet, run_xgboost, compare_models
from database import init_db, save_forecast, load_forecasts, load_forecast_values, delete_forecast

# ─── Конфигурация страницы ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Прогнозирование продаж",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Инициализация БД при первом запуске
init_db()

# ─── Стили ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1E40AF; }
    .metric-label { font-size: 13px; color: #64748B; margin-top: 4px; }
    .section-header { border-left: 4px solid #2563EB; padding-left: 12px; }
</style>
""", unsafe_allow_html=True)


# ─── Боковое меню ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
    st.title("Прогнозирование продаж")
    st.markdown("---")
    page = st.radio(
        "Навигация",
        ["🏠 Главная", "📊 Анализ данных", "🔮 Прогнозирование", "🗂️ История прогнозов"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Датасет: Customer Shopping Dataset\nИсточник: Kaggle\nПериод: 2021–2023")


# ─── Загрузка данных (кэшируем) ───────────────────────────────────────────────
@st.cache_data
def get_data(file):
    return load_data(file)


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 1 — ГЛАВНАЯ
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Главная":
    st.markdown("## 🏠 Главная страница")
    st.markdown("Загрузите CSV-файл датасета, чтобы начать работу.")

    uploaded = st.file_uploader(
        "Выберите файл customer_shopping_data.csv",
        type=["csv"],
        help="Датасет: kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset",
    )

    if uploaded:
        df = get_data(uploaded)
        st.session_state["df"] = df
        stats = get_summary_stats(df)

        st.success(f"✅ Данные загружены успешно! Строк: {len(df):,}")
        st.markdown("### 📋 Сводная статистика")

        cols = st.columns(3)
        metrics_list = [
            ("💰 Суммарная выручка", f"{stats['total_revenue']:,.0f}"),
            ("🛒 Количество заказов", f"{stats['total_orders']:,}"),
            ("📦 Средний чек", f"{stats['avg_order_value']:,.2f}"),
            ("🏷️ Категорий товаров", str(stats["num_categories"])),
            ("🏪 Торговых центров", str(stats["num_malls"])),
            ("📅 Период данных", stats["date_range"]),
        ]
        for i, (label, value) in enumerate(metrics_list):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div><br>
                """, unsafe_allow_html=True)

        st.markdown("### 👀 Предпросмотр данных")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("### 📌 Описание столбцов")
        col_desc = {
            "invoice_no":     "Номер чека",
            "customer_id":    "Идентификатор покупателя",
            "gender":         "Пол покупателя",
            "age":            "Возраст покупателя",
            "category":       "Категория товара",
            "quantity":       "Количество единиц",
            "price":          "Цена за единицу",
            "payment_method": "Способ оплаты",
            "invoice_date":   "Дата покупки",
            "shopping_mall":  "Торговый центр",
            "product_name":   "Название товара (добавлено при предобработке)",
            "revenue":        "Выручка = цена × количество",
        }
        desc_df = pd.DataFrame(
            {"Столбец": list(col_desc.keys()), "Описание": list(col_desc.values())}
        )
        st.dataframe(desc_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Пожалуйста, загрузите CSV-файл датасета.")
        st.markdown("""
        **Как получить датасет:**
        1. Перейдите на [Kaggle](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset)
        2. Нажмите **Download**
        3. Распакуйте архив и загрузите `customer_shopping_data.csv`
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 2 — АНАЛИЗ ДАННЫХ (EDA)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Анализ данных":
    st.markdown("## 📊 Разведочный анализ данных (EDA)")

    if "df" not in st.session_state:
        st.warning("⚠️ Сначала загрузите данные на странице «Главная».")
        st.stop()

    df = st.session_state["df"]

    # ── Фильтры в сайдбаре ──
    with st.sidebar:
        st.markdown("### 🔧 Фильтры")
        years = sorted(df["year"].unique())
        selected_years = st.multiselect("Год", years, default=years)

        categories = sorted(df["category"].unique())
        selected_cats = st.multiselect("Категория", categories, default=categories)

    df_f = df[df["year"].isin(selected_years) & df["category"].isin(selected_cats)]

    if df_f.empty:
        st.error("Нет данных для выбранных фильтров.")
        st.stop()

    # ── Графики ──
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_revenue_over_time(df_f), use_container_width=True)
    with col2:
        st.plotly_chart(plot_heatmap_month_year(df_f), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_category_revenue(df_f), use_container_width=True)
    with col4:
        st.plotly_chart(plot_mall_revenue(df_f), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(plot_payment_pie(df_f), use_container_width=True)
    with col6:
        st.plotly_chart(plot_revenue_by_weekday(df_f), use_container_width=True)

    st.plotly_chart(plot_top_products(df_f, top_n=10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 3 — ПРОГНОЗИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Прогнозирование":
    st.markdown("## 🔮 Прогнозирование продаж")

    if "df" not in st.session_state:
        st.warning("⚠️ Сначала загрузите данные на странице «Главная».")
        st.stop()

    df = st.session_state["df"]
    daily_df = get_daily_sales(df)

    # ── Настройки в сайдбаре ──
    with st.sidebar:
        st.markdown("### ⚙️ Настройки прогноза")
        model_choice = st.selectbox(
            "Метод ИАД",
            ["ARIMA", "Prophet", "XGBoost", "Сравнить все"],
            help="Выберите модель для прогнозирования",
        )
        horizon = st.slider(
            "Горизонт прогноза (дней)", min_value=7, max_value=90, value=30, step=7
        )
        save_result = st.checkbox("Сохранить результат в БД", value=True)

    # ── Описание выбранного метода ──
    method_info = {
        "ARIMA": (
            "**ARIMA** (AutoRegressive Integrated Moving Average) — "
            "классический статистический метод для временных рядов. "
            "Учитывает тренд, автокорреляцию и скользящее среднее ошибок."
        ),
        "Prophet": (
            "**Prophet** (Meta) — современный метод декомпозиции временных рядов. "
            "Автоматически выявляет тренд, годовую и недельную сезонность."
        ),
        "XGBoost": (
            "**XGBoost** — градиентный бустинг (машинное обучение). "
            "Использует лаговые признаки и временные паттерны для прогноза."
        ),
    }
    if model_choice in method_info:
        st.info(method_info[model_choice])
    else:
        st.info("Будут запущены все три модели и показано сравнение метрик качества.")

    # ── Запуск ──
    if st.button("▶️ Запустить прогноз", type="primary", use_container_width=True):
        all_metrics = {}

        if model_choice in ["ARIMA", "Сравнить все"]:
            with st.spinner("Обучение ARIMA..."):
                fc, metrics, fig = run_arima(daily_df, forecast_days=horizon)
            st.markdown("### ARIMA")
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",  f"{metrics['MAE']:,.2f}")
            c2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
            c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            all_metrics["ARIMA"] = metrics
            if save_result:
                fid = save_forecast("ARIMA", metrics, fc, horizon)
                st.success(f"💾 ARIMA сохранена (ID: {fid})")

        if model_choice in ["Prophet", "Сравнить все"]:
            with st.spinner("Обучение Prophet..."):
                fc, metrics, fig = run_prophet(daily_df, forecast_days=horizon)
            st.markdown("### Prophet")
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",  f"{metrics['MAE']:,.2f}")
            c2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
            c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            all_metrics["Prophet"] = metrics
            if save_result:
                fid = save_forecast("Prophet", metrics, fc, horizon)
                st.success(f"💾 Prophet сохранена (ID: {fid})")

        if model_choice in ["XGBoost", "Сравнить все"]:
            with st.spinner("Обучение XGBoost..."):
                fc, metrics, fig = run_xgboost(daily_df, forecast_days=horizon)
            st.markdown("### XGBoost")
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",  f"{metrics['MAE']:,.2f}")
            c2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
            c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            all_metrics["XGBoost"] = metrics
            if save_result:
                fid = save_forecast("XGBoost", metrics, fc, horizon)
                st.success(f"💾 XGBoost сохранена (ID: {fid})")

        # ── Сравнительный график ──
        if len(all_metrics) > 1:
            st.markdown("### 📊 Сравнение моделей")
            st.plotly_chart(compare_models(all_metrics), use_container_width=True)

            st.markdown("**Интерпретация метрик:**")
            st.markdown("""
            - **MAE** (Mean Absolute Error) — средняя ошибка в единицах выручки. Чем меньше — тем лучше.
            - **RMSE** (Root Mean Squared Error) — сильнее штрафует крупные ошибки. Чем меньше — тем лучше.
            - **MAPE** (Mean Absolute Percentage Error) — ошибка в процентах. Значение < 10% считается хорошим.
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 4 — ИСТОРИЯ ПРОГНОЗОВ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ История прогнозов":
    st.markdown("## 🗂️ История сохранённых прогнозов")

    history = load_forecasts()

    if history.empty:
        st.info("Пока нет сохранённых прогнозов. Запустите прогнозирование и сохраните результат.")
    else:
        st.dataframe(
            history.rename(columns={
                "id": "ID", "model_name": "Модель",
                "created_at": "Дата создания", "horizon": "Горизонт (дн.)",
                "mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 🔍 Просмотр прогноза")
        col_sel, col_del = st.columns([3, 1])

        with col_sel:
            selected_id = st.selectbox(
                "Выберите ID прогноза",
                options=history["id"].tolist(),
                format_func=lambda x: f"ID {x} — {history[history['id']==x]['model_name'].values[0]} "
                                       f"({history[history['id']==x]['created_at'].values[0][:10]})",
            )

        with col_del:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Удалить", type="secondary"):
                delete_forecast(selected_id)
                st.success(f"Прогноз ID {selected_id} удалён.")
                st.rerun()

        if selected_id:
            vals = load_forecast_values(selected_id)
            row  = history[history["id"] == selected_id].iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",  f"{row['mae']:,.2f}")
            c2.metric("RMSE", f"{row['rmse']:,.2f}")
            c3.metric("MAPE", f"{row['mape']:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vals["ds"], y=vals["y_hat"],
                mode="lines+markers", name="Прогноз",
                line=dict(color="#EF4444", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=list(vals["ds"]) + list(vals["ds"])[::-1],
                y=list(vals["upper"]) + list(vals["lower"])[::-1],
                fill="toself",
                fillcolor="rgba(239,68,68,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Доверительный интервал",
            ))
            fig.update_layout(
                title=f"Прогноз ID {selected_id} — {row['model_name']}",
                xaxis_title="Дата",
                yaxis_title="Выручка",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Прогнозные значения:**")
            st.dataframe(
                vals.rename(columns={
                    "ds": "Дата", "y_hat": "Прогноз",
                    "lower": "Нижняя граница", "upper": "Верхняя граница"
                }),
                use_container_width=True,
                hide_index=True,
            )
