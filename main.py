import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용해 연도에 따른 기온 변화를 살펴보고, 회귀 직선을 이용해 예상 기온을 계산합니다.")

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 숫자형 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 필요한 데이터만 사용
    df = df.dropna(subset=["날짜", "연도", "평균기온"])

    return df


df = load_data()

# --------------------------------------------------
# 연도별 평균기온 계산
# --------------------------------------------------
yearly = (
    df[df["연도"] <= 2025]
    .groupby("연도")
    .agg(
        관측일수=("날짜", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)

# 관측일수가 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly["연평균기온"] = yearly["연평균기온"].round(2)

# --------------------------------------------------
# 회귀 분석
# 독립 변수: 1908년부터 지난 연수
# --------------------------------------------------
yearly["지난연수"] = yearly["연도"] - 1908

x = yearly["지난연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

# 1차 회귀식 y = ax + b
slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀선용 값
yearly["회귀기온"] = slope * yearly["지난연수"] + intercept

# --------------------------------------------------
# 회귀식 / 정보 표시
# --------------------------------------------------
st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("사용한 연도 수", f"{len(yearly)}년")

with col2:
    st.metric("시작 연도", f"{int(yearly['연도'].min())}년")

with col3:
    st.metric("끝 연도", f"{int(yearly['연도'].max())}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")

st.write(
    f"**회귀식:** 연평균기온 = "
    f"{slope:.4f} × (연도 - 1908) + {intercept:.4f}"
)

# --------------------------------------------------
# 산점도 + 회귀선
# --------------------------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 연도 슬라이더
# --------------------------------------------------
st.subheader("🌡️ 연도별 예상 기온")

selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 1908년부터 지난 연수
selected_elapsed = selected_year - 1908

# 회귀식으로 예상 기온 계산
predicted_temperature = slope * selected_elapsed + intercept

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temperature:.2f} ℃"
)

st.caption(
    "예상 기온은 관측값이 아니라, 300일 이상 관측된 연도의 "
    "연평균기온으로 만든 1차 회귀 직선을 이용한 계산값입니다."
)

# --------------------------------------------------
# 연도별 데이터
# --------------------------------------------------
with st.expander("📋 회귀 분석에 사용된 연도별 데이터 보기"):
    display_df = yearly[
        ["연도", "관측일수", "연평균기온", "회귀기온"]
    ].copy()

    display_df.columns = [
        "연도",
        "관측일수",
        "연평균기온(℃)",
        "회귀기온(℃)"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
