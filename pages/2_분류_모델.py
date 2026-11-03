import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
st.set_page_config(page_title=f"분류 모델 · {APP_TITLE}", page_icon="🤖", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data201/main/stroke.csv"

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}
KOR_TO_COL = {v: k for k, v in KOR.items()}
DEFAULT_KOR = [KOR[f] for f in FEATURES if f != "bmi"]

LOGI = "로지스틱 회귀(확률로 답하는 모델)"
TREE = "의사결정트리(질문으로 답하는 모델)"
BASE = "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델"

COLOR_NO = "#2E86AB"      # 뇌졸중 아님(점)
COLOR_YES = "#C4561A"     # 뇌졸중(점)
TINT_NO = "#E3F0F7"       # 트리가 '아님'이라고 답하는 칸
TINT_YES = "#FBE0D1"      # 트리가 '뇌졸중'이라고 답하는 칸
LEAF_NO = "#CFE6F2"
LEAF_YES = "#F4B183"

st.markdown(
    """
    <style>
    .stApp { background-color: #FFF8F0; }
    .card {
        background: #FFFFFF;
        border: 1px solid #F3D9BE;
        border-left: 8px solid #E8833A;
        border-radius: 14px;
        padding: 20px 18px;
        box-shadow: 0 2px 6px rgba(200, 120, 50, 0.10);
        height: 100%;
    }
    .card .label { font-size: 0.98rem; color: #8A5A33; margin-bottom: 8px; min-height: 2.6em; font-weight: 600; }
    .card .value { font-size: 2.8rem; font-weight: 800; color: #C4561A; line-height: 1.1; }
    .card .cap { font-size: 0.85rem; color: #8A5A33; margin-bottom: 10px; }
    .card .small { display: flex; justify-content: space-between; gap: 8px; font-size: 0.85rem; color: #6B4A2E;
                   border-top: 1px dashed #F3D9BE; padding-top: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt(v: float) -> str:
    return f"{round(float(v), 2):g}"


# ---------------------------------------------------------------
# 제목 + 데이터 읽기
# ---------------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:0'><span style='font-size:1.3em'>🤖</span> 분류 모델</h1>",
    unsafe_allow_html=True,
)
st.caption("stroke가 1이면 뇌졸중, 0이면 아님입니다. 이 페이지에서는 뇌졸중을 양성으로 둡니다.")

try:
    df = load_data(DATA_URL)
except Exception as e:  # noqa: BLE001
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.\n\n({e})")
    st.stop()

# ---------------------------------------------------------------
# 1) 입력 속성 고르기
# ---------------------------------------------------------------
st.subheader("1. 입력으로 쓸 속성 고르기")
chosen_kor = st.multiselect(
    "입력으로 사용할 속성 (두 개 이상)",
    options=[KOR[f] for f in FEATURES],
    default=DEFAULT_KOR,
)
if len(chosen_kor) < 2:
    st.info("속성을 두 개 이상 골라 주세요. 두 개보다 적으면 그림을 그릴 수 없어서 여기서 멈춥니다.")
    st.stop()

feats = [f for f in FEATURES if KOR[f] in chosen_kor]  # 원래 순서대로 정리

# ---------------------------------------------------------------
# 2) 훈련용 / 테스트용 나누기 (번호 순으로 정렬 → 열 명씩 묶어 앞 세 명을 테스트용)
# ---------------------------------------------------------------
data = df.sort_values("id", kind="stable").reset_index(drop=True)
position = np.arange(len(data))
is_test = (position % 10) < 3

train = data[~is_test].copy()
test = data[is_test].copy()
y_train = train["stroke"].to_numpy()
y_test = test["stroke"].to_numpy()

X_train = train[feats].astype(float)
X_test = test[feats].astype(float)

# 체질량지수는 고른 경우에만, 훈련용의 중앙값으로 채운다
if "bmi" in feats:
    bmi_median = X_train["bmi"].median()
    n_fill_train = int(X_train["bmi"].isna().sum())
    n_fill_test = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    fill_note = (
        f" 체질량지수의 빈 값은 훈련용의 중앙값 {bmi_median:.1f}로 채웠습니다 "
        f"(훈련용 {n_fill_train:,}명, 테스트용 {n_fill_test:,}명)."
    )
else:
    fill_note = ""

st.caption(
    f"전체 {len(data):,}명을 번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 정했습니다. "
    f"테스트용 {len(test):,}명, 훈련용 {len(train):,}명입니다." + fill_note
)

# 크기 맞추기는 훈련용으로만 한다
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3) 모델 만들기
# ---------------------------------------------------------------
logi = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)

acc = {
    LOGI: (logi.score(X_train_s, y_train), logi.score(X_test_s, y_test)),
    TREE: (tree.score(X_train, y_train), tree.score(X_test, y_test)),
    BASE: (base.score(X_train_s, y_train), base.score(X_test_s, y_test)),
}

# ---------------------------------------------------------------
# 4) 정확도 카드
# ---------------------------------------------------------------
st.subheader("2. 정확도 비교")
for col, name in zip(st.columns(3), [LOGI, TREE, BASE]):
    tr, te = acc[name]
    col.markdown(
        f"""
        <div class="card">
            <div class="label">{name}</div>
            <div class="value">{pct(te)}</div>
            <div class="cap">테스트 데이터의 정확도</div>
            <div class="small"><span>훈련 데이터 {pct(tr)}</span><span>테스트 데이터 {pct(te)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------
# 5) 두 속성을 골라 그리는 산점도
# ---------------------------------------------------------------
st.subheader("3. 테스트 데이터 산점도와 경계")

c1, c2 = st.columns(2)
with c1:
    x_kor = st.selectbox("가로축", chosen_kor, index=0)
with c2:
    y_options = [k for k in chosen_kor if k != x_kor]
    y_kor = st.selectbox("세로축", y_options, index=0)
fx, fy = KOR_TO_COL[x_kor], KOR_TO_COL[y_kor]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세운다
others = [f for f in feats if f not in (fx, fy)]
medians = {f: float(X_test[f].median()) for f in others}

# 그림의 범위 (테스트 데이터 기준, 5% 여백)
def axis_range(values: pd.Series):
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    return lo - pad, hi + pad


xr = axis_range(X_test[fx])
yr = axis_range(X_test[fy])

# (a) 의사결정트리가 나눈 칸: 격자 위에서 답을 계산해 옅은 색으로 칠한다
n_grid = 150
gx = np.linspace(xr[0], xr[1], n_grid)
gy = np.linspace(yr[0], yr[1], n_grid)
GX, GY = np.meshgrid(gx, gy)
grid = pd.DataFrame({f: np.full(GX.size, medians[f]) for f in feats})
grid[fx] = GX.ravel()
grid[fy] = GY.ravel()
grid = grid[feats]
Z = tree.predict(grid).reshape(GX.shape)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=gx, y=gy, z=Z, zmin=0, zmax=1,
        colorscale=[[0, TINT_NO], [1, TINT_YES]],
        showscale=False, hoverinfo="skip",
    )
)
# 칸 색깔 범례용 (보이지 않는 점)
for label, tint in [("트리의 답: 뇌졸중 아님 칸", TINT_NO), ("트리의 답: 뇌졸중 칸", TINT_YES)]:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14, color=tint, line=dict(color="#B9A58F", width=1)),
            name=label,
        )
    )

# (b) 로지스틱 회귀가 확률 0.5로 가르는 선: w·x + b = 0
w = logi.coef_[0]
b0 = float(logi.intercept_[0])
idx = {f: i for i, f in enumerate(feats)}
ax_ = w[idx[fx]] / scaler.scale_[idx[fx]]
ay_ = w[idx[fy]] / scaler.scale_[idx[fy]]
c_ = b0 - w[idx[fx]] * scaler.mean_[idx[fx]] / scaler.scale_[idx[fx]] \
        - w[idx[fy]] * scaler.mean_[idx[fy]] / scaler.scale_[idx[fy]]
for f in others:
    c_ += w[idx[f]] * (medians[f] - scaler.mean_[idx[f]]) / scaler.scale_[idx[f]]

x_span, y_span = xr[1] - xr[0], yr[1] - yr[0]
line_status = "ok"
if abs(ax_) * x_span < 1e-12 and abs(ay_) * y_span < 1e-12:
    line_status = "none"
else:
    n_line = 600
    if abs(ay_) * y_span >= abs(ax_) * x_span:
        lx = np.linspace(xr[0], xr[1], n_line)
        ly = -(ax_ * lx + c_) / ay_
    else:
        ly = np.linspace(yr[0], yr[1], n_line)
        lx = -(ay_ * ly + c_) / ax_
    eps = 1e-9
    inside = (lx >= xr[0] - eps) & (lx <= xr[1] + eps) & (ly >= yr[0] - eps) & (ly <= yr[1] + eps)
    if not inside.any():
        line_status = "outside"
    else:
        lx = np.where(inside, lx, np.nan)
        ly = np.where(inside, ly, np.nan)
        fig.add_trace(
            go.Scatter(
                x=lx, y=ly, mode="lines", connectgaps=False,
                line=dict(color="#3B2314", width=3),
                name="로지스틱 회귀 0.5 경계선",
            )
        )

# (c) 테스트 데이터 점 (실제 뇌졸중 여부로 색 나누기)
for value, label, color in [(0, "실제: 뇌졸중 아님", COLOR_NO), (1, "실제: 뇌졸중", COLOR_YES)]:
    m = y_test == value
    fig.add_trace(
        go.Scatter(
            x=X_test.loc[m, fx], y=X_test.loc[m, fy], mode="markers",
            marker=dict(size=7, color=color, opacity=0.65, line=dict(color="#FFFFFF", width=0.6)),
            name=label,
        )
    )

fig.update_xaxes(title=x_kor, range=list(xr), showgrid=False)
fig.update_yaxes(title=y_kor, range=list(yr), showgrid=False)
fig.update_layout(
    height=560,
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    legend=dict(title="범례", orientation="v"),
)
st.plotly_chart(fig, use_container_width=True)

# 그림 아래 설명 글
if others:
    med_text = ", ".join(f"{KOR[f]} {fmt(medians[f])}" for f in others)
    st.markdown(
        f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: **{med_text}**"
    )
else:
    st.markdown("고른 속성이 두 개뿐이라, 세워 둔 다른 속성은 없습니다.")

if line_status == "outside":
    st.markdown(
        f"{LOGI}이 확률 0.5로 가르는 선은 이 설정에서 그림 밖에 있어서 그려지지 않았습니다."
    )
elif line_status == "none":
    st.markdown(f"이 설정에서는 {LOGI}의 0.5 경계선을 그릴 수 없습니다.")
else:
    st.markdown(f"검은 선은 {LOGI}이 뇌졸중일 확률을 0.5로 보는 자리입니다.")
st.markdown(
    f"옅은 파란 칸과 옅은 주황 칸은 {TREE}이 각각 ‘뇌졸중 아님’과 ‘뇌졸중’이라고 답하는 자리입니다."
)

# ---------------------------------------------------------------
# 6) 의사결정트리가 던진 질문 (DOT → st.graphviz_chart)
# ---------------------------------------------------------------
st.subheader("4. 의사결정트리가 던진 질문")

t = tree.tree_
path = tree.decision_path(X_train)
n_node = np.asarray(path.sum(axis=0)).ravel().astype(int)
n_pos = np.asarray(path.T.dot(y_train)).ravel().astype(int)


def question_text(node: int) -> str:
    f = feats[t.feature[node]]
    if f in BINARY:
        return f"{KOR[f]}이 있나요?"
    particle = "이" if f == "avg_glucose_level" else "가"
    return f"{KOR[f]}{particle} {t.threshold[node]:.1f} 이하인가?"


dot_lines = [
    "digraph Tree {",
    '  graph [fontname="Noto Sans KR, Malgun Gothic, AppleGothic, sans-serif", nodesep=0.3, ranksep=0.55, bgcolor="transparent"];',
    '  node [shape=box, style="rounded,filled", fontname="Noto Sans KR, Malgun Gothic, AppleGothic, sans-serif", margin="0.18,0.10", width=2.4, color="#B9A58F"];',
    '  edge [fontname="Noto Sans KR, Malgun Gothic, AppleGothic, sans-serif", color="#6B4A2E"];',
]
leaf_answers = []
for i in range(t.node_count):
    n, k = int(n_node[i]), int(n_pos[i])
    rate = k / n * 100 if n else 0.0
    body = f"훈련용 {n:,}명\\n뇌졸중 {k:,}명 ({rate:.1f}%)"
    is_leaf = t.children_left[i] == -1
    if is_leaf:
        ans = int(tree.classes_[int(np.argmax(t.value[i][0]))])
        leaf_answers.append(ans)
        ans_text = "뇌졸중" if ans == 1 else "뇌졸중 아님"
        fill = LEAF_YES if ans == 1 else LEAF_NO
        label = f"답: {ans_text}\\n{body}"
    else:
        fill = "#FFFFFF"
        label = f"{question_text(i)}\\n{body}"
    dot_lines.append(f'  n{i} [label="{label}", fillcolor="{fill}"];')

for i in range(t.node_count):
    if t.children_left[i] == -1:
        continue
    left, right = int(t.children_left[i]), int(t.children_right[i])
    f = feats[t.feature[i]]
    if f in BINARY:   # 왼쪽 가지는 '없음'(0), 오른쪽 가지는 '있음'(1)
        left_label, right_label = "아니요", "예"
    else:             # 왼쪽 가지는 '이하', 오른쪽 가지는 '초과'
        left_label, right_label = "예", "아니요"
    dot_lines.append(f'  n{i} -> n{left} [label="{left_label}"];')
    dot_lines.append(f'  n{i} -> n{right} [label="{right_label}"];')
dot_lines.append("}")

st.graphviz_chart("\n".join(dot_lines), use_container_width=True)

n_leaf = len(leaf_answers)
n_leaf_no = sum(1 for a in leaf_answers if a == 0)
asked = [KOR[feats[j]] for j in sorted(set(int(v) for v in t.feature if v >= 0))]

st.markdown(
    f"답을 내는 마디는 모두 **{n_leaf}칸**이고, 그중 **{n_leaf_no}칸**이 ‘뇌졸중 아님’이라고 답합니다."
)
if asked:
    st.markdown(
        f"고른 속성 {len(feats)}개 가운데 이 나무가 실제로 물은 것은 **{', '.join(asked)}**입니다."
    )
else:
    st.markdown("이 나무는 어떤 속성도 묻지 않고 바로 답을 냈습니다.")
