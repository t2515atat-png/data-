
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/stroke.csv"
)

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인하세요.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 화면 제목 및 소개
# --------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")

st.markdown(
    """
    이 실습실에서는 뇌졸중 관련 데이터를 살펴보고,
    데이터의 특징을 파악하는 것부터 시작합니다.

    데이터에는 개인의 성별, 나이, 건강 관련 정보,
    생활 습관 및 뇌졸중 발생 여부 등이 포함되어 있습니다.
    """
)

st.info(
    "이 화면은 데이터 탐색을 위한 교육용 자료입니다. "
    "개인의 실제 뇌졸중 발생 가능성을 진단하는 용도로 사용할 수 없습니다."
)


# --------------------------------------------------
# 주요 통계 계산
# --------------------------------------------------
total_people = len(df)
total_columns = len(df.columns)

stroke_count = int(
    (pd.to_numeric(df["stroke"], errors="coerce") == 1).sum()
)

stroke_rate = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)


# --------------------------------------------------
# 숫자 카드 4개
# --------------------------------------------------
st.subheader("📊 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="전체 사람 수",
    value=f"{total_people:,}명",
)

col2.metric(
    label="열 개수",
    value=f"{total_columns:,}개",
)

col3.metric(
    label="뇌졸중 발생 기록 (stroke = 1)",
    value=f"{stroke_count:,}명",
)

col4.metric(
    label="뇌졸중 발생 기록 비율",
    value=f"{stroke_rate:.2f}%",
)


# --------------------------------------------------
# 열 정보 표
# --------------------------------------------------
st.divider()
st.subheader("📝 데이터 열 살펴보기")

st.write(
    "각 열의 이름과 값의 종류, 빈 값 개수를 확인해 보세요. "
    "'우리말 뜻'은 교재를 참고해 직접 채워 넣을 수 있도록 비워 두었습니다."
)


def get_value_type(series):
    """자료형과 값의 종류를 간단히 표시"""
    dtype = series.dtype
    unique_values = series.dropna().unique()
    unique_count = len(unique_values)

    if pd.api.types.is_numeric_dtype(dtype):
        return f"숫자형 · 고유값 {unique_count:,}개"

    if unique_count <= 8:
        values = ", ".join(map(str, unique_values))
        return f"문자형 · {values}"

    return f"문자형 · 고유값 {unique_count:,}개"


column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": ["" for _ in df.columns],
    "값의 종류": [
        get_value_type(df[column])
        for column in df.columns
    ],
    "빈 값 개수": [
        int(df[column].isna().sum())
        for column in df.columns
    ],
})

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True,
    height=460,
)

st.caption(
    "고유값은 해당 열에 등장하는 서로 다른 값의 개수입니다. "
    "빈 값 개수는 값이 입력되지 않은 셀의 개수입니다."
)


# --------------------------------------------------
# 원본 데이터 처음 다섯 줄
# --------------------------------------------------
st.divider()
st.subheader("🔍 원본 데이터 처음 5행")

st.write(
    "아래 표는 데이터를 수정하거나 결측값을 채우기 전의 "
    "원본 데이터 중 첫 다섯 행입니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 데이터 출처 입력란
# --------------------------------------------------
st.divider()
st.subheader("📚 데이터 출처")

st.write(
    "교재에 적힌 데이터 출처를 아래 칸에 입력하세요."
)

st.text_area(
    "데이터 출처",
    value="",
    placeholder="교재에 나온 데이터 출처를 여기에 적으세요.",
    height=100,
    key="data_source",
)

st.caption(
    "참고: BMI 열의 빈 값은 원본 그대로 유지합니다. "
    "이 화면에서는 결측값을 임의로 채우거나 삭제하지 않습니다."
)


