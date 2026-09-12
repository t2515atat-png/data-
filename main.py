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
# 그래프 2. 장르 및 영화별 총 관객 수 (트리맵)
# ============================================================
st.divider()
st.header("2. 장르 및 영화별 총 관객 수")

# 총 관객 수가 0보다 큰 데이터만 사용 (트리맵 오류 방지)
df_treemap = df[df["total_audi"] > 0].copy()

fig2 = px.treemap(
    df_treemap,
    path=["genre", "movieNm"],  # 장르 하위에 영화명이 위치하는 계층 구조
    values="total_audi",        # 칸의 크기: 총 관객 수
    title="장르 및 영화별 총 관객 수 분포",
)

# 마우스를 올렸을 때 영화명(또는 장르명)과 총 관객 수 표시
fig2.update_traces(
    hovertemplate="<b>%{label}</b><br>총 관객 수: %{value:,}명<extra></extra>"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("**이 그래프로 알 수 있는 것**")
st.text_input(
    "문구를 입력하세요.",
    placeholder="예: 장르별 총 관객 수의 비중과 특정 장르 내에서 어떤 영화가 흥행했는지 알 수 있다.",
    key="graph2_note",
)

# ============================================================
# 그래프 3. 총 관객 수 분포 (히스토그램)
# ============================================================
st.divider()
st.header("3. 총 관객 수 분포")

fig3 = px.histogram(
    df,
    x="total_audi",
    nbins=30,
    title="영화별 총 관객 수 분포 (히스토그램)",
    labels={"total_audi": "총 관객 수(명)"},
)

fig3.update_layout(
    yaxis_title_text="영화 수",
)

fig3.update_traces(
    hovertemplate="관객 수 구간: %{x}<br>영화 수: %{y}편<extra></extra>"
)

st.plotly_chart(fig3, use_container_width=True)

# 최댓값 영화 및 분포 특징 동적 추출
top_movie = df.loc[df["total_audi"].idxmax()]
top_movie_name = top_movie["movieNm"]
top_movie_audi = top_movie["total_audi"]

st.info(
    f"💡 **데이터 주요 특징**\n\n"
    f"- 대부분의 영화는 관객 수가 상대적으로 적은 **하위 구간(약 100만 명 미만)**에 밀집되어 있습니다.\n"
    f"- 가장 관객 수가 많은 영화는 **{top_movie_name}** (약 {top_movie_audi:,.0f}명)입니다."
)

st.markdown("**이 그래프로 알 수 있는 것**")
st.text_input(
    "문구를 입력하세요.",
    placeholder="예: 박스오피스 상위권 영화 중에서도 대다수의 관객 수 분포와 극소수 흥행작의 편중도를 확인할 수 있다.",
    key="graph3_note",
)
