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

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 연평균기온을 이용하여 장기간의 기온 변화 추세와 "
    "최근 20년의 변화 추세를 비교합니다."
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df["연도"] = df["날짜"].dt.year

    df = df.dropna(subset=["날짜", "연도", "평균기온"])

    return df


df = load_data()

# --------------------------------------------------
# 연도별 평균기온 계산
# 2025년까지 + 관측일수 300일 이상
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

yearly = yearly[
    (yearly["관측일수"] >= 300) &
    (yearly["연도"] >= 1908)
].copy()

yearly["연평균기온"] = yearly["연평균기온"].round(2)

# --------------------------------------------------
# 전체 기간 회귀
# 독립변수 = 1908년부터 지난 연수
# --------------------------------------------------
yearly["지난연수"] = yearly["연도"] - 1908

x_all = yearly["지난연수"].to_numpy()
y_all = yearly["연평균기온"].to_numpy()

slope_all, intercept_all = np.polyfit(x_all, y_all, 1)

# 100년당 상승 기온
slope_all_100 = slope_all * 100

# 전체 기간 상관계수
correlation_all = np.corrcoef(x_all, y_all)[0, 1]

# 전체 회귀선
yearly["전체회귀기온"] = (
    slope_all * yearly["지난연수"] + intercept_all
)

# --------------------------------------------------
# 최근 20년 회귀
# 마지막 사용 연도부터 20년
# --------------------------------------------------
end_year = int(yearly["연도"].max())
recent_start_year = end_year - 19

recent20 = yearly[
    yearly["연도"] >= recent_start_year
].copy()

# 최근 20년의 연도를 독립변수로 사용
x_recent = recent20["연도"].to_numpy()
y_recent = recent20["연평균기온"].to_numpy()

slope_recent, intercept_recent = np.polyfit(
    x_recent,
    y_recent,
    1
)

# 100년당 상승 기온
slope_recent_100 = slope_recent * 100

# 최근 20년 상관계수
correlation_recent = np.corrcoef(
    x_recent,
    y_recent
)[0, 1]

recent20["최근20년회귀기온"] = (
    slope_recent * recent20["연도"] + intercept_recent
)

# --------------------------------------------------
# 기울기 비교
# --------------------------------------------------
st.subheader("🌡️ 100년에 몇 ℃ 변하는가?")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="전체 기간",
        value=f"{slope_all_100:+.2f} ℃ / 100년",
        border=True
    )
    st.caption(
        f"{int(yearly['연도'].min())}년 ~ "
        f"{int(yearly['연도'].max())}년"
    )

with col2:
    st.metric(
        label="최근 20년",
        value=f"{slope_recent_100:+.2f} ℃ / 100년",
        border=True
    )
    st.caption(
        f"{recent_start_year}년 ~ {end_year}년"
    )

st.info(
    "기울기는 회귀직선의 기울기를 100배한 값입니다. "
    "예를 들어 +1.50 ℃/100년은 같은 추세가 100년 동안 "
    "이어진다고 가정했을 때 약 1.50℃ 상승하는 추세라는 뜻입니다."
)

# --------------------------------------------------
# 분석 정보
# --------------------------------------------------
st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "사용한 연도 수",
        f"{len(yearly)}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{int(yearly['연도'].min())}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{int(yearly['연도'].max())}년"
    )

with col4:
    st.metric(
        "전체 상관계수",
        f"{correlation_all:.3f}"
    )

# --------------------------------------------------
# 회귀식 표시
# --------------------------------------------------
st.write(
    f"**전체 기간 회귀식:** "
    f"연평균기온 = {slope_all:.4f} × (연도 - 1908) "
    f"+ {intercept_all:.4f}"
)

st.write(
    f"**최근 20년 회귀식:** "
    f"연평균기온 = {slope_recent:.4f} × 연도 "
    f"+ {intercept_recent:.2f}"
)

# --------------------------------------------------
# 산점도 + 전체 회귀선 + 최근 20년 회귀선
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

# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["전체회귀기온"],
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "전체 회귀기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀선
fig.add_trace(
    go.Scatter(
        x=recent20["연도"],
        y=recent20["최근20년회귀기온"],
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(width=3, dash="dash"),
        hovertemplate=(
            "연도: %{x}년<br>"
            "최근 20년 회귀기온: %{y:.2f}℃"
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
# 상관계수 비교
# --------------------------------------------------
st.subheader("📌 상관계수 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간 상관계수",
        f"{correlation_all:.3f}"
    )

with col2:
    st.metric(
        "최근 20년 상관계수",
        f"{correlation_recent:.3f}"
    )

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

# 전체 기간 회귀식으로 예상
predicted_temperature = (
    slope_all * selected_elapsed + intercept_all
)

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temperature:.2f} ℃"
)

st.caption(
    "예상 기온은 실제 관측값이 아니라 전체 기간의 "
    "회귀직선을 이용하여 계산한 값입니다."
)

# --------------------------------------------------
# 사용 데이터
# --------------------------------------------------
with st.expander("📋 회귀 분석에 사용된 연도별 데이터 보기"):

    display_df = yearly[
        [
            "연도",
            "관측일수",
            "연평균기온",
            "전체회귀기온"
        ]
    ].copy()

    display_df.columns = [
        "연도",
        "관측일수",
        "연평균기온(℃)",
        "전체기간 회귀기온(℃)"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
