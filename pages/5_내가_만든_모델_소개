
import streamlit as st
import pandas as pd


st.title("📝 내가 만든 모델 소개")
st.write(
    "앞에서 탐구한 내용을 직접 정리해 모델 소개서를 완성해 보세요. "
    "작성한 내용은 이 페이지의 화면에만 표시됩니다."
)

# 성능 비교 항목
METRICS = [
    "정확도",
    "재현율",
    "F1",
    "찾아낸 환자",
    "헛짚은 사람",
]

# 이 페이지에서만 사용하는 입력 항목
QUESTIONS = [
    ("intro_data", "1. 무슨 데이터입니까"),
    ("intro_features", "2. 무엇을 보고 맞힙니까"),
    ("intro_changes", "3. 무엇을 고쳤습니까"),
    ("intro_safety", "4. 이 모델을 검진센터가 사용해도 됩니까"),
    ("intro_limits", "5. 이 모델이 못 하는 것 한 가지"),
]

# 입력 내용은 Streamlit 세션 상태에만 둔다.
# 파일 저장, 다운로드, 외부 전송 기능은 제공하지 않는다.
if "intro_metrics_before" not in st.session_state:
    st.session_state["intro_metrics_before"] = {
        metric: "" for metric in METRICS
    }

if "intro_metrics_after" not in st.session_state:
    st.session_state["intro_metrics_after"] = {
        metric: "" for metric in METRICS
    }


def show_answer(value):
    """내용이 비어 있으면 미작성 안내를 표시한다."""
    if value is None or not str(value).strip():
        st.info("아직 적지 않았습니다.")
    else:
        st.write(value)


tab_write, tab_preview = st.tabs(
    ["✏️ 작성하기", "📄 한 장으로 보기"]
)


# ==================================================
# 탭 1: 작성하기
# ==================================================
with tab_write:
    st.subheader("1. 내가 만든 모델 설명")

    st.caption(
        "각 질문에 맞게 여러 문장으로 작성할 수 있습니다. "
        "작성한 내용은 다른 탭에서도 확인할 수 있습니다."
    )

    for key, question in QUESTIONS:
        st.markdown(f"**{question}**")

        st.text_area(
            question,
            key=key,
            height=120,
            placeholder="여기에 내용을 적어 주세요.",
            label_visibility="collapsed",
        )

    st.divider()
    st.subheader("2. 모델 성능 비교표")

    st.write(
        "앞 페이지에서 확인한 값을 직접 옮겨 적으세요. "
        "모델을 고치기 전과 고친 뒤의 결과를 비교하는 표입니다."
    )

    st.caption(
        "백분율은 예를 들어 85.0%처럼 적고, 사람 수는 3명처럼 적어도 됩니다. "
        "아직 값이 없으면 빈칸으로 두세요."
    )

    before_values = {}
    after_values = {}

    for metric in METRICS:
        if metric not in st.session_state["intro_metrics_before"]:
            st.session_state["intro_metrics_before"][metric] = ""
        if metric not in st.session_state["intro_metrics_after"]:
            st.session_state["intro_metrics_after"][metric] = ""

    header1, header2, header3 = st.columns([1.2, 2, 2])
    header1.markdown("**평가 항목**")
    header2.markdown("**고치기 전**")
    header3.markdown("**고친 뒤**")

    for i, metric in enumerate(METRICS):
        col1, col2, col3 = st.columns([1.2, 2, 2])

        with col1:
            st.write(metric)

        with col2:
            st.text_input(
                f"{metric} - 고치기 전",
                key=f"intro_before_{i}",
                value=st.session_state["intro_metrics_before"][metric],
                placeholder="값 입력",
                label_visibility="collapsed",
            )

        with col3:
            st.text_input(
                f"{metric} - 고친 뒤",
                key=f"intro_after_{i}",
                value=st.session_state["intro_metrics_after"][metric],
                placeholder="값 입력",
                label_visibility="collapsed",
            )

    # 위젯에 입력한 값을 세션 상태의 비교표에 반영한다.
    for i, metric in enumerate(METRICS):
        st.session_state["intro_metrics_before"][metric] = (
            st.session_state.get(f"intro_before_{i}", "")
        )
        st.session_state["intro_metrics_after"][metric] = (
            st.session_state.get(f"intro_after_{i}", "")
        )

    st.divider()
    st.caption(
        "작성 내용은 현재 앱 화면의 세션 상태에서만 사용합니다. "
        "이 페이지에는 파일 저장이나 외부 전송 기능이 없습니다."
    )


# ==================================================
# 탭 2: 한 장으로 보기
# ==================================================
with tab_preview:
    st.subheader("내가 만든 모델 소개서")

    st.caption(
        "작성하기 탭에 적은 내용을 순서대로 모아 보여 줍니다. "
        "비어 있는 항목은 미작성으로 표시합니다."
    )

    for number, (key, question) in enumerate(QUESTIONS, start=1):
        st.markdown(f"### {number}. {question}")
        show_answer(st.session_state.get(key, ""))

    st.divider()
    st.markdown("### 6. 모델 성능 비교표")

    preview_rows = []

    for metric in METRICS:
        before_value = st.session_state["intro_metrics_before"].get(
            metric, ""
        )
        after_value = st.session_state["intro_metrics_after"].get(
            metric, ""
        )

        preview_rows.append({
            "평가 항목": metric,
            "고치기 전": (
                before_value.strip()
                if str(before_value).strip()
                else "아직 적지 않았습니다"
            ),
            "고친 뒤": (
                after_value.strip()
                if str(after_value).strip()
                else "아직 적지 않았습니다"
            ),
        })

    preview_table = pd.DataFrame(preview_rows)

    st.dataframe(
        preview_table,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.caption(
        "이 소개서는 작성 내용을 화면에 모아 보여 주는 용도입니다. "
        "내용을 파일로 저장하거나 다른 곳으로 보내지 않습니다."
    )

