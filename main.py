
import streamlit as st
import pandas as pd

# =====================================
# 1. 페이지 설정
# =====================================
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================
# 2. 데이터 주소
# =====================================
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


# =====================================
# 3. 홈 화면
# =====================================
st.title("🧠 뇌졸중 예측 실습실")

st.write(
    "뇌졸중 데이터를 탐색하고 분석하며 "
    "예측 모델을 실습하는 공간입니다."
)

st.divider()

# =====================================
# 4. 데이터 불러오기
# =====================================
st.subheader("📂 데이터 불러오기")

data_url = st.text_input(
    "데이터 출처(URL)",
    value=DEFAULT_DATA_URL,
    key="home_source_url",
)

try:
    df = load_data(data_url.strip())

    if df.empty:
        st.warning("불러온 데이터에 행이 없습니다.")
        st.stop()

except Exception as e:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "URL과 인터넷 연결을 확인하세요."
    )
    st.exception(e)
    st.stop()

# =====================================
# 5. 통계 계산
# =====================================
if "stroke" in df.columns:
    stroke_values = pd.to_numeric(
        df["stroke"], errors="coerce"
    )

    stroke_count = int(
        (stroke_values == 1).sum()
    )

    valid_count = int(
        stroke_values.isin([0, 1]).sum()
    )

    stroke_rate = (
        stroke_count / valid_count * 100
        if valid_count > 0
        else 0
    )
else:
    stroke_count = 0
    stroke_rate = 0

# =====================================
# 6. 요약 카드
# =====================================
st.subheader("📊 데이터 요약")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 인원", f"{len(df):,}명")
col2.metric("전체 열 개수", f"{len(df.columns):,}개")
col3.metric("뇌졸중 경험자", f"{stroke_count:,}명")
col4.metric("뇌졸중 경험 비율", f"{stroke_rate:.2f}%")

st.caption(
    "뇌졸중 경험 비율은 stroke 값이 0 또는 1인 "
    "사람을 기준으로 계산합니다."
)

st.divider()

# =====================================
# 7. 열 정보 표
# =====================================
st.subheader("🗂️ 데이터 열 정보")

st.write(
    "한국어 의미 칸을 클릭하여 각 열의 의미를 입력하세요."
)

column_info = pd.DataFrame({
    "열 이름": df.columns.tolist(),
    "한국어 의미": ["" for _ in df.columns],
    "값의 유형": [
        str(df[col].dtype) for col in df.columns
    ],
    "결측값 개수": [
        int(df[col].isna().sum()) for col in df.columns
    ],
})

st.data_editor(
    column_info,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=[
        "열 이름",
        "값의 유형",
        "결측값 개수",
    ],
    column_config={
        "한국어 의미": st.column_config.TextColumn(
            "한국어 의미",
            help="각 열의 의미를 입력하세요.",
        ),
    },
    key="home_column_editor",
)

st.caption(
    "한국어 의미는 화면에서 편집할 수 있습니다. "
    "별도 저장 기능은 포함되어 있지 않습니다."
)

st.divider()

# =====================================
# 8. 데이터 미리보기
# =====================================
st.subheader("🔍 데이터 미리보기")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.info(
    "왼쪽 사이드바의 '1_탐색' 또는 '탐색' 페이지를 선택해 "
    "데이터 분석 화면으로 이동하세요."
)



