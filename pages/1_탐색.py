
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    # 분석에 사용할 열의 자료형 정리
    for col in [
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "stroke",
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


try:
    df = load_data()
except Exception as e:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "인터넷 연결과 CSV 주소를 확인하세요."
    )
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🔎 데이터 탐색")
st.write(
    "뇌졸중 데이터의 분포와 집단별 차이를 그래프와 표로 살펴봅니다."
)


# --------------------------------------------------
# 1. 나이와 평균 혈당의 분포
# --------------------------------------------------
st.divider()
st.subheader("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    age_data = df[["age"]].dropna()

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
    )

    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    glucose_data = df[["avg_glucose_level"]].dropna()

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수",
        },
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


# --------------------------------------------------
# 2. 뇌졸중 경험 여부에 따른 나이와 혈당 비교
# --------------------------------------------------
st.divider()
st.subheader("2. 뇌졸중 경험 여부에 따른 비교")

stroke_df = df[df["stroke"].isin([0, 1])].copy()

stroke_df["뇌졸중 경험"] = stroke_df["stroke"].map({
    0: "경험 없음",
    1: "경험 있음",
})

col1, col2 = st.columns(2)

with col1:
    age_box_data = stroke_df[
        ["뇌졸중 경험", "age"]
    ].dropna()

    fig_age_box = px.box(
        age_box_data,
        x="뇌졸중 경험",
        y="age",
        color="뇌졸중 경험",
        category_orders={
            "뇌졸중 경험": ["경험 없음", "경험 있음"]
        },
        title="뇌졸중 경험 여부별 나이",
        labels={
            "뇌졸중 경험": "뇌졸중 경험 여부",
            "age": "나이",
        },
    )

    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    glucose_box_data = stroke_df[
        ["뇌졸중 경험", "avg_glucose_level"]
    ].dropna()

    fig_glucose_box = px.box(
        glucose_box_data,
        x="뇌졸중 경험",
        y="avg_glucose_level",
        color="뇌졸중 경험",
        category_orders={
            "뇌졸중 경험": ["경험 없음", "경험 있음"]
        },
        title="뇌졸중 경험 여부별 평균 혈당",
        labels={
            "뇌졸중 경험": "뇌졸중 경험 여부",
            "avg_glucose_level": "평균 혈당",
        },
    )

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 두 집단의 평균값
mean_table = (
    stroke_df.groupby("뇌졸중 경험", observed=True)
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(["경험 없음", "경험 있음"])
    .reset_index()
)

mean_table = mean_table.rename(columns={
    "뇌졸중 경험": "뇌졸중 경험 여부",
    "사람_수": "사람 수",
    "평균_나이": "평균 나이",
    "평균_혈당": "평균 혈당",
})

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.markdown("**두 집단의 평균값 비교**")
st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 3. 고혈압 여부에 따른 뇌졸중 비율
# --------------------------------------------------
st.divider()
st.subheader("3. 고혈압 여부에 따른 뇌졸중 비율")

hypertension_df = df[
    df["hypertension"].isin([0, 1])
    & df["stroke"].isin([0, 1])
].copy()

hypertension_summary = (
    hypertension_df.groupby("hypertension")
    .agg(
        사람_수=("stroke", "size"),
        뇌졸중_사람_수=("stroke", "sum"),
        뇌졸중_비율=("stroke", "mean"),
    )
    .reindex([0, 1])
    .reset_index()
)

hypertension_summary["고혈압 여부"] = (
    hypertension_summary["hypertension"]
    .map({0: "고혈압 없음", 1: "고혈압 있음"})
)

hypertension_summary["뇌졸중 비율(%)"] = (
    hypertension_summary["뇌졸중_비율"] * 100
)

fig_hypertension = px.bar(
    hypertension_summary,
    x="고혈압 여부",
    y="뇌졸중 비율(%)",
    text=hypertension_summary["뇌졸중 비율(%)"].map(
        lambda x: f"{x:.2f}%"
    ),
    title="고혈압 여부별 뇌졸중 비율",
    labels={
        "고혈압 여부": "고혈압",
        "뇌졸중 비율(%)": "뇌졸중 비율(%)",
    },
)

fig_hypertension.update_traces(textposition="outside")
fig_hypertension.update_layout(
    yaxis_title="뇌졸중 비율(%)",
    xaxis_title="",
)

st.plotly_chart(fig_hypertension, use_container_width=True)


# --------------------------------------------------
# 4. 심장병 여부에 따른 뇌졸중 비율
# --------------------------------------------------
st.divider()
st.subheader("4. 심장병 여부에 따른 뇌졸중 비율")

heart_df = df[
    df["heart_disease"].isin([0, 1])
    & df["stroke"].isin([0, 1])
].copy()

heart_summary = (
    heart_df.groupby("heart_disease")
    .agg(
        사람_수=("stroke", "size"),
        뇌졸중_사람_수=("stroke", "sum"),
        뇌졸중_비율=("stroke", "mean"),
    )
    .reindex([0, 1])
    .reset_index()
)

heart_summary["심장병 여부"] = (
    heart_summary["heart_disease"]
    .map({0: "심장병 없음", 1: "심장병 있음"})
)

heart_summary["뇌졸중 비율(%)"] = (
    heart_summary["뇌졸중_비율"] * 100
)

fig_heart = px.bar(
    heart_summary,
    x="심장병 여부",
    y="뇌졸중 비율(%)",
    text=heart_summary["뇌졸중 비율(%)"].map(
        lambda x: f"{x:.2f}%"
    ),
    title="심장병 여부별 뇌졸중 비율",
    labels={
        "심장병 여부": "심장병",
        "뇌졸중 비율(%)": "뇌졸중 비율(%)",
    },
)

fig_heart.update_traces(textposition="outside")
fig_heart.update_layout(
    yaxis_title="뇌졸중 비율(%)",
    xaxis_title="",
)

st.plotly_chart(fig_heart, use_container_width=True)


# --------------------------------------------------
# 5. BMI 결측 여부와 뇌졸중 비율
# --------------------------------------------------
st.divider()
st.subheader("5. BMI 빈 값이 있는 사람들의 뇌졸중 비율")

missing_bmi = df[df["bmi"].isna()].copy()

missing_bmi_count = len(missing_bmi)

missing_bmi_stroke_count = int(
    (
        pd.to_numeric(missing_bmi["stroke"], errors="coerce") == 1
    ).sum()
)

missing_bmi_valid = pd.to_numeric(
    missing_bmi["stroke"], errors="coerce"
).isin([0, 1])

missing_bmi_valid_count = int(missing_bmi_valid.sum())

missing_bmi_rate = (
    missing_bmi_stroke_count / missing_bmi_valid_count * 100
    if missing_bmi_valid_count > 0
    else 0
)

valid_stroke = pd.to_numeric(
    df["stroke"], errors="coerce"
).isin([0, 1])

overall_stroke_count = int(
    (pd.to_numeric(df["stroke"], errors="coerce") == 1).sum()
)

overall_valid_count = int(valid_stroke.sum())

overall_stroke_rate = (
    overall_stroke_count / overall_valid_count * 100
    if overall_valid_count > 0
    else 0
)

bmi_comparison = pd.DataFrame({
    "집단": [
        "BMI 빈 값이 있는 사람",
        "전체 사람",
    ],
    "사람 수": [
        missing_bmi_count,
        len(df),
    ],
    "뇌졸중 경험자 수": [
        missing_bmi_stroke_count,
        overall_stroke_count,
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_rate, 2),
        round(overall_stroke_rate, 2),
    ],
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "뇌졸중 비율은 stroke 값이 0 또는 1로 확인되는 사람을 기준으로 계산합니다."
)


# --------------------------------------------------
# 6. 흡연 상태별 사람 수
# --------------------------------------------------
st.divider()
st.subheader("6. 흡연 상태별 사람 수")

smoking_counts = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "이 페이지는 데이터의 분포와 집단별 차이를 탐색합니다. "
    "관찰된 차이만으로 원인과 결과를 단정할 수는 없습니다."
)
