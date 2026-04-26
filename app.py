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

from data_loader import load_data, get_daily_sales_filtered, get_summary_stats
from eda import (
    plot_revenue_over_time,
    plot_category_revenue,
    plot_mall_revenue,
    plot_payment_pie,
    plot_revenue_by_weekday,
    plot_heatmap_month_year,
    plot_top_products,
)
from models import run_prophet, run_xgboost, compare_models, compare_forecasts_chart
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
        background: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #60A5FA; }
    .metric-label { font-size: 13px; opacity: 0.75; margin-top: 4px; }
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
    st.caption("Информационная система прогнозирования продаж\nна основе методов интеллектуального анализа данных")


# ─── Загрузка данных (кэшируем) ───────────────────────────────────────────────
@st.cache_data
def get_data(file):
    return load_data(file)


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 1 — ГЛАВНАЯ
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Главная":
    st.markdown("## 🏠 Главная страница")
    st.markdown(
        "Информационная система прогнозирования продаж компании "
        "с применением инструментов интеллектуального анализа данных. "
        "Загрузите CSV-файл с историей продаж вашей компании, чтобы начать работу."
    )

    uploaded = st.file_uploader(
        "Загрузите CSV-файл с данными о продажах",
        type=["csv"],
        help="Файл должен содержать столбцы с датой покупки, ценой, количеством, "
             "категорией товара и точкой продаж (см. требования ниже).",
    )

    if uploaded:
        df = get_data(uploaded)
        st.session_state["df"] = df
        stats = get_summary_stats(df)

        st.success(f"✅ Данные загружены успешно! Строк: {len(df):,}")
        st.markdown("### 📋 Сводная статистика")

        cols = st.columns(3)
        metrics_list = [
            ("💰 Суммарная выручка", f"{stats['total_revenue']:,.0f} ₽"),
            ("🛒 Количество заказов", f"{stats['total_orders']:,}"),
            ("📦 Средний чек", f"{stats['avg_order_value']:,.2f} ₽"),
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
            "invoice_no":     "Номер чека / документа продажи",
            "customer_id":    "Идентификатор покупателя",
            "gender":         "Пол покупателя",
            "age":            "Возраст покупателя",
            "category":       "Категория товара",
            "quantity":       "Количество единиц",
            "price":          "Цена за единицу, ₽",
            "payment_method": "Способ оплаты",
            "invoice_date":   "Дата покупки",
            "shopping_mall":  "Точка продаж (магазин, филиал, ТЦ)",
            "product_name":   "Название товара (добавлено при предобработке)",
            "revenue":        "Выручка = цена × количество, ₽",
        }
        desc_df = pd.DataFrame(
            {"Столбец": list(col_desc.keys()), "Описание": list(col_desc.values())}
        )
        st.dataframe(desc_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Пожалуйста, загрузите CSV-файл с историей продаж.")
        st.markdown("""
        **Требования к файлу:**

        CSV-файл должен содержать следующие столбцы (регистр и пробелы не важны):

        | Столбец | Тип | Назначение |
        |---|---|---|
        | `invoice_date` | дата | Дата продажи (формат `ДД/ММ/ГГГГ`) — **обязательно** |
        | `price` | число | Цена за единицу — **обязательно** |
        | `quantity` | число | Количество единиц — **обязательно** |
        | `category` | текст | Категория товара — для аналитики и фильтров |
        | `shopping_mall` | текст | Точка продаж (магазин/филиал) — для аналитики и фильтров |
        | `payment_method` | текст | Способ оплаты — для EDA |
        | `invoice_no`, `customer_id`, `gender`, `age` | — | Опционально |

        Для построения прогноза достаточно минимум **60 дней истории продаж**.
        Чем длиннее история — тем точнее будет прогноз.

        💡 *Для демонстрации работы системы можно использовать открытый датасет
        [Customer Shopping Dataset](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset) с Kaggle.*
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

    # ── Настройки в сайдбаре ──
    with st.sidebar:
        st.markdown("### ⚙️ Настройки прогноза")
        model_choice = st.selectbox(
            "Метод ИАД",
            ["Prophet", "XGBoost", "Сравнить обе"],
        )
        horizon = st.slider(
            "Горизонт прогноза (дней)", min_value=7, max_value=90, value=30, step=7
        )
        st.markdown("### 🔍 Фильтры данных")
        category_options = ["Все категории"] + sorted(df["category"].unique().tolist())
        mall_options     = ["Все ТЦ"] + sorted(df["shopping_mall"].unique().tolist())
        selected_category = st.selectbox("Категория товара", category_options)
        selected_mall     = st.selectbox("Торговый центр",   mall_options)
        save_result = st.checkbox("Сохранить результат в БД", value=True)

    daily_df = get_daily_sales_filtered(df, selected_category, selected_mall)

    # Проверка достаточности данных после фильтрации
    if len(daily_df) < 60:
        st.error("Недостаточно данных для выбранного фильтра (менее 60 дней). Выберите другую категорию или ТЦ.")
        st.stop()

    filter_label = ""
    if selected_category != "Все категории":
        filter_label += f" · {selected_category}"
    if selected_mall != "Все ТЦ":
        filter_label += f" · {selected_mall}"
    if filter_label:
        st.caption(f"Фильтр:{filter_label} · {len(daily_df)} дней данных")

    # ── Описание выбранного метода ──
    method_info = {
        "Prophet": (
            "**Prophet** (Meta) — статистический метод декомпозиции временного ряда "
            "на тренд (кусочно-линейный) и сезонные компоненты (годовая и недельная). "
            "Не требует feature engineering — работает напрямую с парой (дата, значение). "
            "Сильная сторона — устойчивая экстраполяция тренда и сезонности."
        ),
        "XGBoost": (
            "**XGBoost** — градиентный бустинг на деревьях решений (машинное обучение). "
            "Превращает временной ряд в таблицу признаков (лаги 1/7/14/30 дней, "
            "скользящие средние 7/30, день недели, месяц) и обучается прогнозировать продажи. "
            "Сильная сторона — улавливает нелинейные взаимодействия между признаками."
        ),
    }
    if model_choice in method_info:
        st.info(method_info[model_choice])
    else:
        st.info("Будут запущены обе модели и показано сравнение метрик и прогнозов "
                "(статистический метод vs машинное обучение).")

    # ── Объяснение метрик качества ──
    with st.expander("ℹ️ Что означают метрики качества (MAE, RMSE, MAPE)"):
        st.markdown("""
После обучения модель проверяется на **20% данных, которые она не видела** при обучении
(тестовая выборка). Прогноз модели сравнивается с фактом, и считаются три метрики ошибки:

#### 📏 MAE — Mean Absolute Error (средняя абсолютная ошибка)

$$ MAE = \\frac{1}{n} \\sum |y_{факт} - y_{прогноз}| $$

Среднее отклонение прогноза от факта **в рублях**.
Например, MAE = 22 500 ₽ означает: в среднем модель ошибается на 22 500 ₽ в день.
*Чем меньше — тем лучше.*

#### 📐 RMSE — Root Mean Squared Error (корень из средней квадратичной ошибки)

$$ RMSE = \\sqrt{\\frac{1}{n} \\sum (y_{факт} - y_{прогноз})^2} $$

Похож на MAE, но **крупные ошибки штрафуются сильнее** (возводятся в квадрат).
Если RMSE значительно больше MAE — у модели есть редкие, но крупные промахи (выбросы).
*Чем меньше — тем лучше.*

#### 📊 MAPE — Mean Absolute Percentage Error (средняя процентная ошибка)

$$ MAPE = \\frac{1}{n} \\sum \\left| \\frac{y_{факт} - y_{прогноз}}{y_{факт}} \\right| \\times 100\\% $$

Универсальная метрика — ошибка в процентах, не зависит от единиц измерения.
Позволяет сравнивать модели на разных датасетах.

**Шкала качества (по Льюису, 1982):**

| MAPE | Качество прогноза |
|---|---|
| 🟢 < 10% | **Отлично** — высокая точность |
| 🟡 10–20% | **Хорошо** — удовлетворительная точность |
| 🟠 20–30% | **Удовлетворительно** — приемлемо для нестабильных рядов |
| 🔴 30–50% | **Слабо** — модель требует доработки |
| ⛔ > 50% | **Неточно** — прогноз непригоден |

---

💡 **Важно понимать:** «реалистично выглядящий» прогноз (с колебаниями) часто
даёт **большую** ошибку, чем гладкая линия около среднего. Метрики измеряют
**точность попадания в конкретный день**, а не визуальную «похожесть» на
исторический ряд. Прогноз с колебаниями верно угадывает амплитуду продаж,
но обычно ошибается в фазе (когда именно будет подъём) — это даёт двойную
ошибку: и в день фактического подъёма, и в день прогнозного.
        """)

    # ── Вспомогательные функции ──
    def mape_badge(mape: float) -> str:
        if mape < 10:
            color, label = "#16A34A", "Отлично"
        elif mape < 20:
            color, label = "#65A30D", "Хорошо"
        elif mape < 30:
            color, label = "#D97706", "Удовлетворительно"
        elif mape < 50:
            color, label = "#EA580C", "Слабо"
        else:
            color, label = "#DC2626", "Неточно"
        return (
            f'<span style="background:{color};color:white;padding:3px 10px;'
            f'border-radius:5px;font-size:13px;font-weight:600">{label} ({mape:.2f}%)</span>'
        )

    def show_metrics(metrics: dict):
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (средняя ошибка)",      f"{metrics['MAE']:,.2f} ₽")
        c2.metric("RMSE (штраф за выбросы)",   f"{metrics['RMSE']:,.2f} ₽")
        with c3:
            st.metric("MAPE (ошибка в %)", "")
            st.markdown(mape_badge(metrics["MAPE"]), unsafe_allow_html=True)

    def show_forecast_conclusion(history_df: pd.DataFrame, forecast_df: pd.DataFrame,
                                  model_name: str, horizon: int) -> None:
        """Текстовый блок с выводом по результатам прогноза продаж."""
        hist_avg = history_df.tail(30)["y"].mean()
        fc_avg   = forecast_df["y_hat"].mean()
        fc_total = forecast_df["y_hat"].sum()
        delta_pct = ((fc_avg - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0.0

        if delta_pct > 5:
            trend_word, trend_emoji = "роста", "📈"
            trend_color, trend_bg = "#38BDF8", "rgba(56, 189, 248, 0.12)"
        elif delta_pct < -5:
            trend_word, trend_emoji = "снижения", "📉"
            trend_color, trend_bg = "#6366F1", "rgba(99, 102, 241, 0.12)"
        else:
            trend_word, trend_emoji = "стабильности", "➡️"
            trend_color, trend_bg = "#3B82F6", "rgba(59, 130, 246, 0.12)"

        peak_row = forecast_df.loc[forecast_df["y_hat"].idxmax()]
        low_row  = forecast_df.loc[forecast_df["y_hat"].idxmin()]
        peak_weekday = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"][peak_row["ds"].weekday()]

        st.markdown(f"""
        <div style="background:{trend_bg};border-left:4px solid {trend_color};
                    padding:14px 18px;border-radius:6px;margin-top:8px;margin-bottom:14px">
            <div style="font-size:15px;font-weight:600;margin-bottom:8px;color:{trend_color}">
                {trend_emoji} Вывод по прогнозу продаж — {model_name}
            </div>
            <div style="font-size:14px;line-height:1.7">
                На горизонте <b>{horizon} дн.</b> модель прогнозирует <b>тенденцию {trend_word} продаж</b>:
                среднесуточные продажи изменятся на <b style="color:{trend_color}">{delta_pct:+.1f}%</b>
                относительно последних 30 дней истории.<br>
                • <b>Среднесуточные продажи:</b> {fc_avg:,.0f} ₽ <span style="opacity:0.65">(история: {hist_avg:,.0f} ₽)</span><br>
                • <b>Суммарные продажи за период:</b> {fc_total:,.0f} ₽<br>
                • <b>Пик продаж</b> ожидается {peak_row['ds'].strftime('%d.%m.%Y')} ({peak_weekday}) — {peak_row['y_hat']:,.0f} ₽<br>
                • <b>Минимум продаж:</b> {low_row['ds'].strftime('%d.%m.%Y')} — {low_row['y_hat']:,.0f} ₽
            </div>
        </div>
        """, unsafe_allow_html=True)

    def export_button(fc_df: pd.DataFrame, model_name: str):
        csv = fc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Скачать прогноз CSV",
            data=csv,
            file_name=f"forecast_{model_name.replace(' ', '_')}_{horizon}d.csv",
            mime="text/csv",
        )

    # ── Запуск ──
    if st.button("▶️ Запустить прогноз", type="primary", use_container_width=True):
        all_metrics   = {}
        all_forecasts = {}

        if model_choice in ["XGBoost", "Сравнить обе"]:
            with st.spinner("Обучение XGBoost..."):
                fc, metrics, fig = run_xgboost(daily_df, forecast_days=horizon)
            st.markdown("### XGBoost")
            st.plotly_chart(fig, use_container_width=True)
            show_metrics(metrics)
            show_forecast_conclusion(daily_df, fc, "XGBoost", horizon)
            export_button(fc, "XGBoost")
            all_metrics["XGBoost"]   = metrics
            all_forecasts["XGBoost"] = fc
            if save_result:
                fid = save_forecast("XGBoost", metrics, fc, horizon)
                st.success(f"💾 XGBoost сохранена (ID: {fid})")

        if model_choice in ["Prophet", "Сравнить обе"]:
            with st.spinner("Обучение Prophet..."):
                fc, metrics, fig = run_prophet(daily_df, forecast_days=horizon)
            st.markdown("### Prophet")
            st.plotly_chart(fig, use_container_width=True)
            show_metrics(metrics)
            show_forecast_conclusion(daily_df, fc, "Prophet", horizon)
            export_button(fc, "Prophet")
            all_metrics["Prophet"]   = metrics
            all_forecasts["Prophet"] = fc
            if save_result:
                fid = save_forecast("Prophet", metrics, fc, horizon)
                st.success(f"💾 Prophet сохранена (ID: {fid})")

        # ── Сравнение моделей ──
        if len(all_metrics) > 1:
            st.markdown("### 📊 Сравнение моделей")

            # Последние 60 дней истории для контекста на графике
            history_tail = daily_df.tail(60)
            st.plotly_chart(
                compare_forecasts_chart(
                    all_forecasts,
                    history_ds=history_tail["ds"],
                    history_y=history_tail["y"],
                ),
                use_container_width=True,
            )

            st.plotly_chart(compare_models(all_metrics), use_container_width=True)

            st.markdown("**Краткая интерпретация:**")
            st.markdown("""
            - **MAE** — средняя ошибка в единицах выручки. Чем меньше — тем лучше.
            - **RMSE** — сильнее штрафует крупные выбросы. Чем меньше — тем лучше.
            - **MAPE** — ошибка в %. 🟢 <10% отлично · 🟡 10–20% хорошо · 🟠 20–30% удовл. · 🔴 30–50% слабо · ⛔ >50% неточно.

            *Подробное объяснение метрик — в блоке «ℹ️ Что означают метрики качества» вверху страницы.*
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
            c1.metric("MAE (средняя ошибка)",    f"{row['mae']:,.2f} ₽")
            c2.metric("RMSE (штраф за выбросы)", f"{row['rmse']:,.2f} ₽")
            c3.metric("MAPE (ошибка в %)",       f"{row['mape']:.2f}%")

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
                yaxis_title="Выручка, ₽",
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
