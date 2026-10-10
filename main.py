
import streamlit as st
import pandas as pd

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# 페이지 등록 및 사이드바 메뉴
# -----------------------------
home_page = st.Page(
    "main.py",
    title="홈",
    icon="🏠",
    default=True,
)

explore_page = st.Page(
    "pages/1_탐색.py",
    title="데이터 탐색",
    icon="🔎",
)

navigation = st.navigation(
    {
        "메뉴": [home_page, explore_page],
    },
    position="sidebar",
)

# 메뉴를 실제로 실행
# 홈 페이지일 때 아래 코드가 표시됨
if navigation == home_page:

    DEFAULT_DATA_URL = (
        "https://raw.githubusercontent.com/"
        "greatsong/modudata/main/data/stroke.csv"
    )

    @st.cache_data
    def load_data(url):
        return pd.read_csv(url, encoding="utf-8")

    # -----------------------------
    # 제목
    # -----------------------------
    st.title("🧠 뇌졸중 예측 실습실")
    st.write(
        "뇌졸중 데이터를 탐색하고 분석하며 "
        "예측 모델을 실습하는 공간입니다."
    )

    st.divider()

    # -----------------------------
    # 데이터 출처 입력
    # -----------------------------
    st.subheader("📂 데이터 불러오기")

    data_url = st.text_input(
        "데이터 출처(URL)",
        value=DEFAULT_DATA_URL,
        key="data_url",
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

    # -----------------------------
    # 뇌졸중 통계 계산
    # -----------------------------
    if "stroke" in df.columns:
        stroke_values = pd.to_numeric(
            df["stroke"], errors="coerce"
        )

        stroke_count = int((stroke_values == 1).sum())
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

    # -----------------------------
    # 요약 카드 4개
    # -----------------------------
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

    # -----------------------------
    # 열 정보 표
    # -----------------------------
    st.subheader("🗂️ 데이터 열 정보")
    st.write(
        "한국어 의미 칸을 클릭해 각 열의 의미를 입력할 수 있습니다."
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
        key="column_info_editor",
    )

    st.caption(
        "한국어 의미는 화면에서 편집할 수 있습니다. "
        "별도 저장 기능이 없으면 새로 실행할 때 유지되지 않을 수 있습니다."
    )

    st.divider()

    # -----------------------------
    # 데이터 미리보기
    # -----------------------------
    st.subheader("🔍 데이터 미리보기")
    st.write("원본 데이터의 처음 5개 행입니다.")

    st.dataframe(
        df.head(5),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.info(
        "왼쪽 사이드바에서 '데이터 탐색'을 선택하면 "
        "나이와 혈당의 분포 및 집단별 차이를 살펴볼 수 있습니다."
    )

# 페이지 표시
navigation.run()

