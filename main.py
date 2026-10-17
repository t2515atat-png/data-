
import streamlit as st
import pandas as pd

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


def show_home():
    st.title("🧠 뇌졸중 예측 실습실")
    st.write(
        "뇌졸중 데이터를 탐색하고 분류 모델의 성능을 평가하는 실습실입니다."
    )

    try:
        df = load_data(DATA_URL)
    except Exception as e:
        st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인하세요.")
        st.exception(e)
        return

    st.subheader("📊 데이터 요약")

    if "stroke" in df.columns:
        stroke = pd.to_numeric(df["stroke"], errors="coerce")
        valid = stroke.isin([0, 1])
        positive_count = int((stroke == 1).sum())
        rate = float(stroke[valid].mean() * 100) if valid.any() else 0
    else:
        positive_count = 0
        rate = 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 인원", f"{len(df):,}명")
    c2.metric("전체 열 개수", f"{len(df.columns):,}개")
    c3.metric("뇌졸중 경험자", f"{positive_count:,}명")
    c4.metric("뇌졸중 경험 비율", f"{rate:.2f}%")

    st.divider()
    st.subheader("🗂️ 데이터 열 정보")

    info = pd.DataFrame({
        "열 이름": df.columns.tolist(),
        "한국어 의미": ["" for _ in df.columns],
        "값의 유형": [str(df[col].dtype) for col in df.columns],
        "결측값 개수": [int(df[col].isna().sum()) for col in df.columns],
    })

    st.data_editor(
        info,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["열 이름", "값의 유형", "결측값 개수"],
        key="home_column_editor",
    )

    st.caption("한국어 의미는 현재 화면에서 편집할 수 있습니다.")

    st.divider()
    st.subheader("🔍 데이터 미리보기")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)


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

model_page = st.Page(
    "pages/2_분류_모델.py",
    title="분류 모델",
    icon="🌳",
)

score_page = st.Page(
    "pages/3_채점표.py",
    title="채점표",
    icon="📋",
)

preprocess_page = st.Page(
    "pages/4_전처리.py",
    title="전처리",
    icon="🧹",
)

navigation = st.navigation(
    [
        home_page,
        explore_page,
        model_page,
        score_page,
        preprocess_page,
    ],
    position="sidebar",
)

navigation.run()
