
import streamlit as st
import pandas as pd

# 페이지 기본 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

# 데이터 주소
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

# 데이터 불러오기
@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


# 제목
st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "뇌졸중 데이터를 탐색하고 분석하며 "
    "예측 모델을 실습하는 공간입니다."
)

st.divider()

# 데이터 불러오기
st.subheader("📂 데이터 불러오기")

data_url = st.text_input(
    "데이터 출처(URL)",
    value=DEFAULT_DATA_URL,
    help="CSV 파일의 URL을 입력하세요.",
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


# 뇌졸중 열 확인 및 숫자 변환
stroke_values = None

if "stroke" in df.columns:
    stroke_values = pd.to_numeric(
        df["stroke"], errors="coerce"
    )

    stroke_count = int((stroke_values == 1).sum())
    valid_stroke_count = int(
        stroke_values.isin([0, 1]).sum()
    )

    stroke_rate = (
        stroke_count / valid_stroke_count * 100
        if valid_stroke_count > 0
        else 0
    )
else:
    stroke_count = 0
    valid_stroke_count = 0
    stroke_rate = 0


# 상단 요약 카드 4개
st.subheader("📊 데이터 요약")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="전체 인원",
        value=f"{len(df):,}명",
    )

with col2:
    st.metric(
        label="전체 열 개수",
        value=f"{len(df.columns):,}개",
    )

with col3:
    st.metric(
        label="뇌졸중 경험자",
        value=f"{stroke_count:,}명",
    )

with col4:
    st.metric(
        label="뇌졸중 경험 비율",
        value=f"{stroke_rate:.2f}%",
    )

st.caption(
    "뇌졸중 경험 비율은 stroke 값이 0 또는 1인 "
    "사람을 기준으로 계산합니다."
)

st.divider()


# 열 정보 표
st.subheader("🗂️ 데이터 열 정보")
st.write(
    "각 열의 의미를 확인하고, "
    "한국어 의미 칸은 직접 수정할 수 있습니다."
)

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "한국어 의미": ["" for _ in df.columns],
    "값의 유형": [
        str(df[col].dtype) for col in df.columns
    ],
    "결측값 개수": [
        int(df[col].isna().sum()) for col in df.columns
    ],
})

edited_info = st.data_editor(
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
            help="각 열이 무엇을 의미하는지 입력하세요.",
        ),
        "값의 유형": st.column_config.TextColumn(
            "값의 유형"
        ),
        "결측값 개수": st.column_config.NumberColumn(
            "결측값 개수",
            format="%d",
        ),
    },
    key="column_info_editor",
)

st.caption(
    "한국어 의미 칸에서 입력한 내용은 현재 화면에서 "
    "확인할 수 있습니다. 별도 파일로 저장되지는 않습니다."
)

st.divider()


# 데이터 미리보기
st.subheader("🔍 데이터 미리보기")
st.write("원본 데이터의 처음 5개 행입니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)

st.divider()


# 페이지 안내
st.subheader("📑 다른 페이지")
st.write(
    "왼쪽 사이드바의 **탐색** 메뉴를 누르면 "
    "나이와 혈당의 분포, 뇌졸중 경험 여부별 비교, "
    "고혈압 및 심장병 여부별 비율 등을 확인할 수 있습니다."
)
