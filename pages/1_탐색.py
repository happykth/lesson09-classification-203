import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
st.set_page_config(page_title=f"탐색 · {APP_TITLE}", page_icon="🔎", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data201/main/stroke.csv"

WARM = ["#E8833A", "#C4561A", "#F2B134", "#8A5A33"]
GROUP_COLORS = {"겪지 않음": "#F2B134", "겪음": "#C4561A"}

st.markdown(
    """
    <style>
    .stApp { background-color: #FFF8F0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        title_font_size=16,
    )
    return fig


# ---------------------------------------------------------------
# 제목 + 데이터 읽기
# ---------------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:0'><span style='font-size:1.3em'>🔎</span> 데이터 탐색</h1>",
    unsafe_allow_html=True,
)
st.caption("뇌졸중 데이터를 그림과 표로 살펴봅니다.")

try:
    df = load_data(DATA_URL)
except Exception as e:  # noqa: BLE001
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.\n\n({e})")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})

# ---------------------------------------------------------------
# 1) 나이와 평균 혈당의 분포 (히스토그램 두 개)
# ---------------------------------------------------------------
st.subheader("1. 나이와 평균 혈당의 분포")
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(
        df, x="age", nbins=30, title="나이 분포",
        labels={"age": "나이(세)", "count": "사람 수"},
        color_discrete_sequence=[WARM[0]],
    )
    fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(style_fig(fig), use_container_width=True)
with c2:
    fig = px.histogram(
        df, x="avg_glucose_level", nbins=40, title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당(mg/dL)"},
        color_discrete_sequence=[WARM[1]],
    )
    fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(style_fig(fig), use_container_width=True)

# ---------------------------------------------------------------
# 2) 뇌졸중 여부별 상자그림 + 평균 표
# ---------------------------------------------------------------
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
order = ["겪지 않음", "겪음"]
c1, c2 = st.columns(2)
with c1:
    fig = px.box(
        df, x="뇌졸중 여부", y="age", color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": order},
        color_discrete_map=GROUP_COLORS,
        title="나이 비교", labels={"age": "나이(세)"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)
with c2:
    fig = px.box(
        df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": order},
        color_discrete_map=GROUP_COLORS,
        title="평균 혈당 비교", labels={"avg_glucose_level": "평균 혈당(mg/dL)"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)

mean_table = (
    df.groupby("뇌졸중 여부")
    .agg(
        사람_수=("stroke", "size"),
        나이_평균=("age", "mean"),
        평균_혈당_평균=("avg_glucose_level", "mean"),
    )
    .reindex(order)
    .reset_index()
    .rename(
        columns={
            "뇌졸중 여부": "뇌졸중",
            "사람_수": "사람 수",
            "나이_평균": "나이 평균(세)",
            "평균_혈당_평균": "평균 혈당 평균(mg/dL)",
        }
    )
)
st.markdown("**두 그룹의 평균값**")
st.dataframe(
    mean_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "사람 수": st.column_config.NumberColumn("사람 수", format="%d"),
        "나이 평균(세)": st.column_config.NumberColumn("나이 평균(세)", format="%.2f"),
        "평균 혈당 평균(mg/dL)": st.column_config.NumberColumn("평균 혈당 평균(mg/dL)", format="%.2f"),
    },
)

# ---------------------------------------------------------------
# 3) 고혈압·심장병 여부별 뇌졸중 비율 (막대그래프)
# ---------------------------------------------------------------
st.subheader("3. 고혈압·심장병 여부별 뇌졸중 비율")


def rate_bar(column: str, title: str):
    rate = (
        df.groupby(column)["stroke"].mean().mul(100).reset_index(name="뇌졸중 비율(%)")
    )
    rate["구분"] = rate[column].map({0: "없음", 1: "있음"})
    fig = px.bar(
        rate, x="구분", y="뇌졸중 비율(%)", color="구분",
        category_orders={"구분": ["없음", "있음"]},
        color_discrete_map={"없음": "#F2B134", "있음": "#C4561A"},
        text=rate["뇌졸중 비율(%)"].map(lambda v: f"{v:.2f}%"),
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)", xaxis_title=None)
    fig.update_yaxes(range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.25, 1)])
    return style_fig(fig)


c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(rate_bar("hypertension", "고혈압 여부별 뇌졸중 비율"), use_container_width=True)
with c2:
    st.plotly_chart(rate_bar("heart_disease", "심장병 여부별 뇌졸중 비율"), use_container_width=True)

# ---------------------------------------------------------------
# 4) bmi가 빈 사람들의 뇌졸중 비율 vs 전체 뇌졸중 비율
# ---------------------------------------------------------------
st.subheader("4. 체질량지수(bmi)가 빈 사람들")
missing = df[df["bmi"].isna()]
n_missing = len(missing)
n_missing_stroke = int((missing["stroke"] == 1).sum())
n_all = len(df)
n_all_stroke = int((df["stroke"] == 1).sum())

bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [n_missing, n_all],
        "뇌졸중 사람 수": [n_missing_stroke, n_all_stroke],
        "뇌졸중 비율(%)": [
            n_missing_stroke / n_missing * 100 if n_missing else float("nan"),
            n_all_stroke / n_all * 100 if n_all else float("nan"),
        ],
    }
)
st.dataframe(
    bmi_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "사람 수": st.column_config.NumberColumn("사람 수", format="%d"),
        "뇌졸중 사람 수": st.column_config.NumberColumn("뇌졸중 사람 수", format="%d"),
        "뇌졸중 비율(%)": st.column_config.NumberColumn("뇌졸중 비율(%)", format="%.2f"),
    },
)

# ---------------------------------------------------------------
# 5) 흡연 상태별 사람 수
# ---------------------------------------------------------------
st.subheader("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .fillna("(빈 값)")
    .value_counts()
    .rename_axis("smoking_status 값")
    .reset_index(name="사람 수")
)
st.dataframe(
    smoking,
    hide_index=True,
    use_container_width=True,
    column_config={"사람 수": st.column_config.NumberColumn("사람 수", format="%d")},
)
