import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="영화 데이터 그래프 도감 2 - 분포와 관계", layout="wide")
st.title("영화 데이터 그래프 도감 2 - 분포와 관계")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # 개봉일: 여덟 자리 숫자(YYYYMMDD) -> 실제 날짜
    df["openDt"] = pd.to_datetime(df["openDt"].astype(str), format="%Y%m%d", errors="coerce")

    # 장르가 여러 개면 첫 번째 장르만 사용
    df["genre"] = (
        df["genre"]
        .fillna("미상")
        .astype(str)
        .str.split(r"[|/]")
        .str[0]
        .str.strip()
        .replace("", "미상")
    )

    return df

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# ============================================================
# 그래프 1. 장르별 영화 편수
# ============================================================
st.divider()
st.header("1. 장르별 영화 편수")

# 장르별 영화 수를 계산
genre_counts = (
    df["genre"]
    .value_counts()
    .rename_axis("장르")
    .reset_index(name="편수")
)

fig1 = px.pie(
    genre_counts,
    names="장르",
    values="편수",
    hole=0.55,
    title="장르별 영화 편수",
)

# 마우스를 올렸을 때 편수와 비율 표시
fig1.update_traces(
    textinfo="none",
    hovertemplate="장르: %{label}<br>편수: %{value}편<br>비율: %{percent}<extra></extra>",
)

fig1.update_layout(
    legend_title_text="장르",
)

st.plotly_chart(fig1, use_container_width=True)

st.markdown("**이 그래프로 알 수 있는 것**")
st.text_input(
    "문구를 입력하세요.",
    placeholder="예: 이 기간에는 어떤 장르의 영화가 가장 많이 개봉했는지 알 수 있다.",
    key="graph1_note",
)

# ============================================================
# 그래프 2. 여기에 다음 그래프를 추가하세요.
# ============================================================
st.divider()
st.header("2. 다음 그래프")
st.info("앞으로 추가할 그래프를 이 구역에 넣을 수 있습니다.")
