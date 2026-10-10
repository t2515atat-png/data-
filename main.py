
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
# 3. 사이드바
# =====================================
with st.sidebar:
    st.title("🧠 뇌졸중 예측 실습실")
    st.caption("데이터 분석 실습")

    st.divider()

    st.write("🏠 홈")
    st.caption("현재 메인 페이지입니다.")

    st.divider()

    st.write("🔎 데이터 탐색")
    st.caption("나이, 혈당, 고혈압, 심장병 분석")

    st.info(
        "다른 페이지로 이동하려면 "
        "사이드바의 페이지 목록에서 "
        "'탐색'을 선택하세요."
    )

# =====================================
# 4. 제목
# =====================================
st.title("🧠 뇌졸중 예측 실습실")

st.write(
    "뇌졸중 데이터를 살펴보고 "
    "데이터 분석과 예측 모델을 실습하는 공간입니다."
)

st.divider()

# =====================================
# 5. 데이터 불러오기
# =====================================
st.subheader("📂 데이터 불러오기")

data_url = st.text_input(
    "데이터 출처(URL)",
    DEFAULT_DATA_URL,
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
# 6. 요약 통계
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
# 7. 열 정보 및 한국어 의미 편집
# =====================================
st.subheader("🗂️ 데이터 열 정보")

st.write(
    "한국어 의미 열의 빈칸을 클릭해 "
    "각 변수의 의미를 입력하세요."
)

column_info = pd.DataFrame({
    "열 이름": df.columns.tolist(),
    "한국어 의미": ["" for _ in df.columns],
    "값의 유형": [
        str(df[col].dtype)
        for col in df.columns
    ],
    "결측값 개수": [
        int(df[col].isna().sum())
        for col in df.columns
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
        "열 이름": st.column_config.TextColumn(
            "열 이름"
        ),
        "한국어 의미": st.column_config.TextColumn(
            "한국어 의미",
            help="변수의 뜻을 직접 입력하세요.",
        ),
        "값의 유형": st.column_config.TextColumn(
            "값의 유형"
        ),
        "결측값 개수": st.column_config.NumberColumn(
            "결측값 개수",
            format="%d",
        ),
    },
)

st.caption(
    "입력한 한국어 의미는 화면에서 수정할 수 있습니다. "
    "현재 코드는 입력 내용을 별도 파일에 저장하지 않습니다."
)

st.divider()

# =====================================
# 8. 데이터 미리보기
# =====================================
st.subheader("🔍 데이터 미리보기")

st.write("원본 데이터의 처음 5개 행입니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.success(
    "데이터를 불러왔습니다. "
    "데이터 탐색 페이지에서 변수별 분포와 "
    "집단 간 차이를 확인해 보세요."
)

