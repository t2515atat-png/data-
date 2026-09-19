import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 관객 수와 흥행 지속 정도를 바탕으로 "
    "비슷한 특성을 가진 영화들을 세 가지 유형으로 나눕니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/kobis_movies.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# =========================================================
# 필요한 열 숫자형 변환
# =========================================================
numeric_columns = [
    "first_scrn",
    "first_show",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =========================================================
# 전체 영화 편수
# =========================================================
total_movies = len(df)


# =========================================================
# 분석용 변수 만들기
# =========================================================
# 원래 값
df["스크린수_원래"] = df["first_scrn"]
df["누적관객_원래"] = df["total_audi"]
df["10위권일수_원래"] = df["days_in_top10"]

# 로그 변환
# log10을 사용하기 위해 0 이하 값은 결측 처리
df["스크린수"] = np.where(
    df["first_scrn"] > 0,
    np.log10(df["first_scrn"]),
    np.nan
)

df["누적관객"] = np.where(
    df["total_audi"] > 0,
    np.log10(df["total_audi"]),
    np.nan
)

# 롱런 지수
df["롱런지수"] = np.where(
    df["first_week_audi"] > 0,
    df["total_audi"] / df["first_week_audi"],
    np.nan
)

# 20 초과는 20으로 제한
df["롱런지수"] = df["롱런지수"].clip(upper=20)


# =========================================================
# 분석에 사용할 네 가지 변수
# =========================================================
feature_columns = [
    "스크린수",
    "누적관객",
    "10위권일수",
    "롱런지수"
]

feature_labels = {
    "스크린수": "스크린 수",
    "누적관객": "누적 관객",
    "10위권일수": "10위권 일수",
    "롱런지수": "롱런 지수"
}


# =========================================================
# 결측값 제거
# =========================================================
analysis_df = df.dropna(
    subset=feature_columns + ["first_week_audi"]
).copy()

# first_week_audi가 0인 영화 제거
analysis_df = analysis_df[
    analysis_df["first_week_audi"] > 0
].copy()

analysis_df = analysis_df.reset_index(drop=True)


# =========================================================
# 전체 편수 / 묶은 편수 표시
# =========================================================
st.markdown(
    f"**전체 영화: {total_movies}편　|　묶음에 사용한 영화: "
    f"{len(analysis_df)}편**"
)


# =========================================================
# 사용할 속성 선택
# =========================================================
st.subheader("⚙️ 묶는 데 사용할 속성")

selected_features = st.multiselect(
    "k-평균에 사용할 속성을 두 개 이상 선택하세요.",
    options=feature_columns,
    default=feature_columns,
    format_func=lambda x: feature_labels[x]
)

if len(selected_features) < 2:
    st.warning("묶는 데 사용할 속성을 최소 두 개 선택해야 합니다.")
    st.stop()


# =========================================================
# 표준화 + KMeans
# =========================================================
X = analysis_df[selected_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

raw_cluster = kmeans.fit_predict(X_scaled)

analysis_df["원래묶음"] = raw_cluster


# =========================================================
# 누적 관객 평균이 큰 순서로 묶음 번호 재지정
# ㉮ → ㉯ → ㉰
# =========================================================
cluster_mean = (
    analysis_df
    .groupby("원래묶음")["누적관객_원래"]
    .mean()
    .sort_values(ascending=False)
)

cluster_order = cluster_mean.index.tolist()

cluster_symbol = {
    cluster_order[0]: "㉮",
    cluster_order[1]: "㉯",
    cluster_order[2]: "㉰"
}

analysis_df["묶음"] = analysis_df["원래묶음"].map(
    cluster_symbol
)


# =========================================================
# 묶음별 색상용 순서
# =========================================================
symbol_order = ["㉮", "㉯", "㉰"]


# =========================================================
# 2차원 산점도
# =========================================================
st.subheader("📊 2차원 산점도")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "가로축",
        options=feature_columns,
        index=0,
        format_func=lambda x: feature_labels[x]
    )

with col2:
    y_feature = st.selectbox(
        "세로축",
        options=feature_columns,
        index=1,
        format_func=lambda x: feature_labels[x]
    )


plot_df = analysis_df.copy()

fig2d = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="묶음",
    category_orders={
        "묶음": symbol_order
    },
    hover_name="movieNm",
    hover_data={
        x_feature: ":.2f",
        y_feature: ":.2f",
        "묶음": True
    },
    labels={
        x_feature: feature_labels[x_feature],
        y_feature: feature_labels[y_feature],
        "묶음": "영화 유형"
    }
)

fig2d.update_traces(
    marker=dict(size=8)
)

fig2d.update_layout(
    height=600,
    legend_title="영화 유형"
)

st.plotly_chart(
    fig2d,
    use_container_width=True
)


# =========================================================
# 3차원 산점도
# =========================================================
st.subheader("📊 3차원 산점도")

if len(selected_features) < 3:

    st.info(
        "묶는 데 사용할 속성을 세 개 이상 선택하면 "
        "3차원 산점도를 표시할 수 있습니다."
    )

else:

    col1, col2, col3 = st.columns(3)

    with col1:
        z_x = st.selectbox(
            "3D 가로축 (X)",
            options=feature_columns,
            index=0,
            format_func=lambda x: feature_labels[x],
            key="3d_x"
        )

    with col2:
        z_y = st.selectbox(
            "3D 세로축 (Y)",
            options=feature_columns,
            index=1,
            format_func=lambda x: feature_labels[x],
            key="3d_y"
        )

    with col3:
        z_z = st.selectbox(
            "3D 높이축 (Z)",
            options=feature_columns,
            index=2,
            format_func=lambda x: feature_labels[x],
            key="3d_z"
        )

    fig3d = px.scatter_3d(
        plot_df,
        x=z_x,
        y=z_y,
        z=z_z,
        color="묶음",
        category_orders={
            "묶음": symbol_order
        },
        hover_name="movieNm",
        hover_data={
            z_x: ":.2f",
            z_y: ":.2f",
            z_z: ":.2f",
            "묶음": True
        },
        labels={
            z_x: feature_labels[z_x],
            z_y: feature_labels[z_y],
            z_z: feature_labels[z_z],
            "묶음": "영화 유형"
        }
    )

    fig3d.update_traces(
        marker=dict(size=4)
    )

    fig3d.update_layout(
        height=700,
        legend_title="영화 유형"
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True
    )


# =========================================================
# 묶음별 통계
# =========================================================
st.subheader("📋 묶음별 특징")

summary_rows = []

for symbol in symbol_order:

    group = analysis_df[
        analysis_df["묶음"] == symbol
    ]

    summary_rows.append({
        "묶음": symbol,
        "영화 편수": len(group),
        "스크린 수 평균": group["스크린수_원래"].mean(),
        "누적 관객 평균": group["누적관객_원래"].mean(),
        "10위권 일수 평균": group["10위권일수_원래"].mean(),
        "롱런 지수 평균": group["롱런지수"].mean()
    })


summary_df = pd.DataFrame(summary_rows)

summary_df["스크린 수 평균"] = (
    summary_df["스크린 수 평균"].round(1)
)

summary_df["누적 관객 평균"] = (
    summary_df["누적 관객 평균"].round(0)
)

summary_df["10위권 일수 평균"] = (
    summary_df["10위권 일수 평균"].round(1)
)

summary_df["롱런 지수 평균"] = (
    summary_df["롱런 지수 평균"].round(2)
)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 묶음별 누적 관객 TOP 5
# =========================================================
st.subheader("🎬 묶음별 누적 관객 TOP 5")

top5_cols = st.columns(3)

for i, symbol in enumerate(symbol_order):

    group = analysis_df[
        analysis_df["묶음"] == symbol
    ].sort_values(
        "누적관객_원래",
        ascending=False
    ).head(5)

    with top5_cols[i]:

        st.markdown(f"### {symbol}")

        if len(group) == 0:

            st.write("영화가 없습니다.")

        else:

            for rank, (_, row) in enumerate(
                group.iterrows(),
                start=1
            ):

                st.write(
                    f"**{rank}. {row['movieNm']}**  \n"
                    f"누적 관객: {row['누적관객_원래']:,.0f}명"
                )
