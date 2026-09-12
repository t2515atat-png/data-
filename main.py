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
# 그래프 2. 일관객 합계 TOP 5 영화 비교
# ============================================================
st.divider()
st.header("2. 일관객 합계 TOP 5 영화의 날짜별 변화")

# 이 기간 동안 일관객 합계가 가장 큰 영화 5편을 계산
top5_movies = (
    df.groupby("영화명")["일관객"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)

top5_df = df[df["영화명"].isin(top5_movies)].sort_values(["날짜", "영화명"])

fig2 = px.line(
    top5_df,
    x="날짜",
    y="일관객",
    color="영화명",
    markers=True,
    title="일관객 합계가 가장 큰 5편의 날짜별 일관객 변화",
    labels={
        "날짜": "날짜",
        "일관객": "일관객 수",
        "영화명": "영화",
    },
    hover_data={
        "날짜": "|%Y-%m-%d",
        "일관객": ":,",
        "영화명": True,
    },
)

fig2.update_traces(
    hovertemplate="영화: %{fullData.name}<br>"
                  "날짜: %{x|%Y-%m-%d}<br>"
                  "일관객: %{y:,}명<extra></extra>"
)

fig2.update_layout(
    hovermode="closest",
    xaxis_title="날짜",
    yaxis_title="일관객 수(명)",
    legend_title="영화",
)

# 범례를 클릭하면 해당 영화의 선을 켜고 끌 수 있음
st.plotly_chart(fig2, use_container_width=True)

st.markdown("**이 그래프로 알 수 있는 것**")
st.text_input(
    "문구를 입력하세요.",
    placeholder="예: 기간 전체에서 관객 수가 많았던 5편의 날짜별 관객 변화 양상을 비교할 수 있다.",
    key="graph2_note",
)


# ============================================================
# 그래프 3. 날짜별 10위권 일관객 합계
# ============================================================
st.divider()
st.header("3. 날짜별 10위권 일관객 합계")

# 날짜별로 그날의 10위권 영화 일관객을 모두 합산
daily_total = (
    df.groupby("날짜", as_index=False)["일관객"]
    .sum()
    .sort_values("날짜")
)

# 합계가 가장 큰 날짜 3일
top3_days = daily_total.nlargest(3, "일관객").copy()

fig3 = px.area(
    daily_total,
    x="날짜",
    y="일관객",
    markers=True,
    title="날짜별 10위권 일관객 합계",
    labels={
        "날짜": "날짜",
        "일관객": "10위권 일관객 합계",
    },
    hover_data={
        "날짜": "|%Y-%m-%d",
        "일관객": ":,",
    },
)

fig3.update_traces(
    hovertemplate="날짜: %{x|%Y-%m-%d}<br>10위권 일관객 합계: %{y:,}명<extra></extra>"
)

# 합계가 가장 큰 3일을 그래프 위에 날짜와 함께 표시
for _, row in top3_days.iterrows():
    fig3.add_annotation(
        x=row["날짜"],
        y=row["일관객"],
        text=f"{row['날짜']:%Y-%m-%d}<br>{row['일관객']:,}명",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-45,
    )

fig3.update_layout(
    hovermode="x unified",
    xaxis_title="날짜",
    yaxis_title="10위권 일관객 합계(명)",
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("**이 그래프로 알 수 있는 것**")
st.text_input(
    "문구를 입력하세요.",
    placeholder="예: 날짜에 따라 영화 상위 10편의 전체 관객 규모가 어떻게 달라지는지 알 수 있다.",
    key="graph3_note",
)

# ============================================================
# 그래프 4. 영화별 일관객 TOP 10
# ============================================================
st.divider()
st.header("4. 영화별 일관객 TOP 10")

# 영화별 이 기간 일관객 합계와 10위권에 든 날수를 계산
# 변수명은 파이썬 문법에 맞게 영문으로 만든 뒤, 화면에 사용할 열 이름으로 변경
movie_summary = (
    df.groupby("영화명")
    .agg(
        total_audience=("일관객", "sum"),
        top10_days=("날짜", "nunique"),
    )
    .rename(columns={
        "total_audience": "일관객합계",
        "top10_days": "10위권_든_날수",
    })
    .sort_values("일관객합계", ascending=False)
    .head(10)
    .sort_values("일관객합계", ascending=True)
    .reset_index()
)

fig4 = px.bar(
    movie_summary,
    x="일관객합계",
    y="영화명",
    orientation="h",
    text="일관객합계",
    title="영화별 이 기간 일관객 TOP 10",
    labels={
        "영화명": "영화명",
        "일관객합계": "일관객 합계",
    },
    custom_data=["10위권_든_날수"],
)

fig4.update_traces(
    texttemplate="%{text:,}명",
    textposition="outside",
    hovertemplate=(
        "영화명: %{y}<br>"
        "이 기간 일관객 합계: %{x:,}명<br>"
        "10위권에 든 날수: %{customdata[0]}일"
        "<extra></extra>"
    ),
)

fig4.update_layout(
    xaxis_title="이 기간 일관객 합계(명)",
    yaxis_title="영화명",
    yaxis={"categoryorder": "array", "categoryarray": movie_summary["영화명"].tolist()},
)

st.plotly_chart(fig4, use_container_width=True)

st.markdown("**이 그래프로 알 수 있는 것**")
st.text_input(
    "문구를 입력하세요.",
    placeholder="예: 이 기간 동안 누적 일관객이 가장 많았던 영화와 10위권에 오래 머문 영화를 비교할 수 있다.",
    key="graph4_note",
)

# ============================================================
# 그래프 5. 여기에 다음 그래프를 추가하세요.
# ============================================================
st.divider()
st.header("5. 다음 그래프")
st.info("앞으로 추가할 그래프를 이 구역에 넣을 수 있습니다.")
