

import streamlit as st
import pandas as pd

# =====================================
# 1. 기본 설정
# =====================================
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


# =====================================
# 2. 홈 화면
# =====================================
def show_home():
    st.title("🧠 뇌졸중 예측 실습실")
    st.write(
        "뇌졸중 데이터를 탐색하고 분석하며 "
        "예측 모델을 실습하는 공간입니다."
    )

    st.divider()
    st.subheader("📂 데이터 불러오기")

    url = st.text_input(
        "데이터 출처(URL)",
        value=DATA_URL,
        key="home_data_url",
    )

    try:
        df = load_data(url.strip())
    except Exception as e:
        st.error(
            "데이터를 불러오지 못했습니다. "
            "URL과 인터넷 연결을 확인하세요."
        )
        st.exception(e)
        return

    if df.empty:
        st.warning("데이터에 행이 없습니다.")
        return

    # 뇌졸중 통계
    if "stroke" in df.columns:
        stroke = pd.to_numeric(
            df["stroke"], errors="coerce"
        )
        stroke_count = int((stroke == 1).sum())
        valid_count = int(stroke.isin([0, 1]).sum())
        stroke_rate = (
            stroke_count / valid_count * 100
            if valid_count else 0
        )
    else:
        stroke_count = 0
        stroke_rate = 0

    # 요약 카드
    st.divider()
    st.subheader("📊 데이터 요약")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 인원", f"{len(df):,}명")
    c2.metric("전체 열 개수", f"{len(df.columns):,}개")
    c3.metric("뇌졸중 경험자", f"{stroke_count:,}명")
    c4.metric("뇌졸중 경험 비율", f"{stroke_rate:.2f}%")

    st.caption(
        "경험 비율은 stroke 값이 0 또는 1인 "
        "사람을 기준으로 계산합니다."
    )

    # 열 정보
    st.divider()
    st.subheader("🗂️ 데이터 열 정보")
    st.write("한국어 의미 칸을 직접 수정할 수 있습니다.")

    info = pd.DataFrame({
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
        info,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=[
            "열 이름",
            "값의 유형",
            "결측값 개수",
        ],
        key="home_column_editor",
    )

    st.caption(
        "수정한 한국어 의미는 현재 화면에서 편집할 수 있으며, "
        "별도 저장 기능은 없습니다."
    )

    # 데이터 미리보기
    st.divider()
    st.subheader("🔍 데이터 미리보기")
    st.dataframe(
        df.head(5),
        use_container_width=True,
        hide_index=True,
    )


# =====================================
# 3. 실제 페이지 등록
# =====================================
home_page = st.Page(
    show_home,
    title="홈",
    icon="🏠",
    default=True,
)

explore_page = st.Page(
    "pages/1_탐색.py",
    title="데이터 탐색",
    icon="🔎",
)

# 사이드바 메뉴 생성
navigation = st.navigation(
    [home_page, explore_page],
    position="sidebar",
)

# 선택한 페이지 실행
navigation.run()


