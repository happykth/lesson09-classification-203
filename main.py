import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 기본 설정 (브라우저 탭 제목과 아이콘)
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data203/main/stroke.csv"
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "데이터_소개.json"

# ---------------------------------------------------------------
# 따뜻한 느낌의 스타일
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #FFF8F0; }
    .card {
        background: #FFFFFF;
        border: 1px solid #F3D9BE;
        border-left: 8px solid #E8833A;
        border-radius: 14px;
        padding: 22px 20px;
        box-shadow: 0 2px 6px rgba(200, 120, 50, 0.10);
    }
    .card .label { font-size: 1.0rem; color: #8A5A33; margin-bottom: 6px; }
    .card .value { font-size: 2.6rem; font-weight: 800; color: #C4561A; line-height: 1.1; }
    .card .unit { font-size: 1.1rem; font-weight: 600; color: #8A5A33; margin-left: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


def describe_values(series: pd.Series) -> str:
    """열의 값의 종류를 짧은 글로 설명한다."""
    values = series.dropna().unique()
    if len(values) <= 10:
        shown = sorted(values.tolist(), key=lambda v: str(v))
        return ", ".join(str(v) for v in shown)
    if pd.api.types.is_numeric_dtype(series):
        return f"숫자 ({len(values)}가지, {series.min():g} ~ {series.max():g})"
    return f"글자 ({len(values)}가지)"


# ---------------------------------------------------------------
# 저장된 기록 읽기 / 쓰기 (내_기록 폴더의 json 파일)
# ---------------------------------------------------------------
def load_notes() -> dict:
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        meanings = data.get("meanings", {})
        source = data.get("source", "")
        if not isinstance(meanings, dict) or not isinstance(source, str):
            raise ValueError("형식이 올바르지 않습니다.")
        return {"meanings": meanings, "source": source}
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {"meanings": {}, "source": ""}


def save_notes(meanings: dict, source: str) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"meanings": meanings, "source": source}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------
# 화면 맨 위: 아이콘 + 제목
# ---------------------------------------------------------------
st.markdown(
    f"<h1 style='margin-bottom:0'><span style='font-size:1.3em'>{APP_ICON}</span> {APP_TITLE}</h1>",
    unsafe_allow_html=True,
)
st.caption("첫 화면: 우리가 다룰 데이터가 무엇인지 알아봅니다.")

try:
    df = load_data(DATA_URL)
except Exception as e:  # noqa: BLE001
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.\n\n({e})")
    st.stop()

notes = load_notes()

# ---------------------------------------------------------------
# 1) 큰 숫자 카드 네 개
# ---------------------------------------------------------------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0.0

cards = [
    ("전체 사람 수", f"{total:,}", "명"),
    ("열 개수", f"{n_cols:,}", "개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}", "명"),
    ("stroke가 1인 비율", f"{ratio:.2f}", "%"),
]
for col, (label, value, unit) in zip(st.columns(4), cards):
    col.markdown(
        f"""
        <div class="card">
            <div class="label">{label}</div>
            <div><span class="value">{value}</span><span class="unit">{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------
# 2) 열 설명 표 + 3) 처음 다섯 줄 + 4) 데이터 출처
#    (우리말 뜻과 출처는 폼으로 묶어 '저장' 버튼으로 한 번에 저장)
# ---------------------------------------------------------------
st.subheader("📋 열 설명 표")
st.caption("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적어 주세요. 다 적은 뒤 맨 아래 [저장] 버튼을 누르면 됩니다.")

info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [notes["meanings"].get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

with st.form("intro_form"):
    edited = st.data_editor(
        info,
        hide_index=True,
        use_container_width=True,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (직접 적기)"),
            "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
        },
        key="meaning_editor",
    )

    st.subheader("👀 데이터 처음 다섯 줄")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)

    st.subheader("📚 데이터 출처")
    source_text = st.text_area(
        "교재에 나온 출처를 적어 주세요.",
        value=notes["source"],
        height=120,
        placeholder="여기에 데이터 출처를 적어 주세요.",
    )

    saved = st.form_submit_button("💾 저장", type="primary")

if saved:
    meanings = {
        str(row["열 이름"]): str(row["우리말 뜻"] or "").strip()
        for _, row in edited.iterrows()
    }
    try:
        save_notes(meanings, source_text.strip())
        st.success("저장했습니다. 새로고침해도 다시 불러옵니다.")
    except OSError as e:
        st.error(f"저장하지 못했습니다: {e}")
