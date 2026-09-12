import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="영화 데이터 그래프 도감 1 - 시간",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)

    # 날짜 열을 실제 날짜(datetime)로 변환
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")

    # 그래프에 사용할 수 있도록 숫자형 열을 정리
    numeric_cols = ["순위", "일관객", "누적관객", "스크린수", "상영횟수"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["날짜", "순위"]).reset_index(drop=True)


st.title("🎬 영화 데이터 그래프 도감 1 - 시간")
st.write("KOBIS 일별 박스오피스 데이터를 이용해 영화의 시간에 따른 관객 변화를 살펴봅니다.")

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 문제가 발생했습니다.")
    st.exception(e)
    st.stop()


# ============================================================
# 그래프 1. 영화별 날짜에 따른 일관객 변화
# 앞으로 그래프를 추가할 때 이 구역 아래에 새로운 섹션을 추가하세요.
# ============================================================
st.header("1. 영화별 날짜에 따른 일관객 변화")

movie_list = sorted(df["영화명"].dropna().unique())

selected_movie = st.selectbox(
    "영화를 선택하세요.",
    movie_list,
)

movie_df = df[df["영화명"] == selected_movie].sort_values("날짜")

fig = px.line(
    movie_df,
    x="날짜",
    y="일관객",
    markers=True,
    title=f"「{selected_movie}」 날짜별 일관객 변화",
    labels={
        "날짜": "날짜",
        "일관객": "일관객 수",
    },
    hover_data={
        "날짜": "|%Y-%m-%d",
        "일관객": ":,",
    },
)

fig.update_traces(
    hovertemplate="날짜: %{x|%Y-%m-%d}<br>일관객: %{y:,}명<extra></extra>"
)

fig.update_layout(
    hovermode="x unified",
    xaxis_title="날짜",
    yaxis_title="일관객 수(명)",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("**이 그래프로 알 수 있는 것**")
st.text_input(
    "문구를 입력하세요.",
    placeholder="예: 영화의 개봉 후 날짜에 따라 일관객 수가 어떻게 변하는지 알 수 있다.",
    key="graph1_note",
)

# ============================================================
# 그래프 2. 여기에 다음 그래프를 추가하세요.
# ============================================================
st.divider()
st.header("2. 다음 그래프")
st.info("앞으로 추가할 그래프를 이 구역에 넣을 수 있습니다.")
