
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# =====================================
# 1. 기본 설정
# =====================================
st.title("📋 뇌졸중 모델 채점표")
st.write(
    "채점용 데이터로 세 모델의 혼동행렬과 "
    "정확도, 정밀도, 재현율, F1을 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "BMI",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

RANDOM_STATE = 42


@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


# =====================================
# 2. 데이터 읽기 및 번호 순 정렬
# =====================================
try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인하세요.")
    st.exception(e)
    st.stop()

required_columns = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke",
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

if "id" in df.columns:
    df = df.sort_values("id", kind="stable").reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")
df = df[df["stroke"].isin([0, 1])].copy()
df["stroke"] = df["stroke"].astype(int)

if len(df) < 4:
    st.error("채점용 3명과 훈련용 데이터를 나눌 수 있도록 데이터가 더 필요합니다.")
    st.stop()

# 앞의 세 명은 채점용, 나머지는 훈련용
test_df = df.iloc[:3].copy()
train_df = df.iloc[3:].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error(
        "훈련용 데이터에 뇌졸중 경험자와 비경험자가 모두 있어야 "
        "분류 모델을 학습할 수 있습니다."
    )
    st.stop()


# =====================================
# 3. 입력 속성 선택
# =====================================
st.subheader("1. 입력 속성 선택")

selected_features = st.multiselect(
    "분류 모델에 사용할 속성을 선택하세요.",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x],
    key="classification_features",
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해 주세요.")
    st.stop()

st.caption(
    "분류 모델 페이지와 동일하게 번호 순으로 정렬한 뒤 "
    "앞의 3명을 채점용으로 고정합니다."
)

# =====================================
# 4. 결측값 처리
# =====================================
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

for col in selected_features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # BMI를 선택한 경우에만 훈련용 BMI 중앙값으로 채움
    if col == "bmi":
        bmi_median = X_train[col].median()
        if pd.isna(bmi_median):
            st.error("훈련용 BMI 중앙값을 계산할 수 없습니다.")
            st.stop()

        X_train[col] = X_train[col].fillna(bmi_median)
        X_test[col] = X_test[col].fillna(bmi_median)

    # 다른 선택 속성에 결측값이 있으면 훈련용 중앙값 사용
    if X_train[col].isna().any():
        median_value = X_train[col].median()

        if pd.isna(median_value):
            st.error(f"'{FEATURE_LABELS[col]}'의 훈련용 값이 모두 비어 있습니다.")
            st.stop()

        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)

# 크기 맞추기는 훈련용 데이터만으로 학습
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================
# 5. 두 분류 모델 학습
# =====================================
# 확률 모델: 로지스틱 회귀
probability_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=2000,
)
probability_model.fit(X_train_scaled, y_train)

positive_index = list(probability_model.classes_).index(1)
probability_values = probability_model.predict_proba(X_test_scaled)[
    :, positive_index
]
probability_predictions = (probability_values >= 0.5).astype(int)

# 질문 모델: 깊이 3, 마지막 마디 최소 5명
question_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
question_model.fit(X_train, y_train)
question_predictions = question_model.predict(X_test).astype(int)

# 기준 모델: 입력을 보지 않고 훈련용에서 많은 쪽으로만 예측
majority_class = int(y_train.value_counts().idxmax())
baseline_predictions = np.full(len(y_test), majority_class, dtype=int)

predictions_by_model = {
    "확률 모델": probability_predictions,
    "질문 모델": question_predictions,
    "다수 집단만 예측": baseline_predictions,
}


# =====================================
# 6. 채점용 데이터 요약
# =====================================
st.divider()
st.subheader("2. 채점용 데이터")

positive_test_count = int((y_test == 1).sum())

st.write(
    f"채점용 사람 수: **{len(test_df)}명**　|　"
    f"그중 실제 뇌졸중인 사람: **{positive_test_count}명**"
)


# =====================================
# 7. 혼동행렬 계산 함수
# =====================================
def get_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 행: 실제 결과 / 열: 예측 결과
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    return tn, fp, fn, tp


def make_confusion_table(y_true, y_pred):
    tn, fp, fn, tp = get_counts(y_true, y_pred)

    return pd.DataFrame(
        [
            [
                f"TN: {tn}명",
                f"FP: {fp}명",
                f"{tn + fp}명",
            ],
            [
                f"FN: {fn}명",
                f"TP: {tp}명",
                f"{fn + tp}명",
            ],
            [
                f"{tn + fn}명",
                f"{fp + tp}명",
                f"{tn + fp + fn + tp}명",
            ],
        ],
        index=[
            "실제: 뇌졸중 아님",
            "실제: 뇌졸중",
            "열 합계",
        ],
        columns=[
            "예측: 뇌졸중 아님",
            "예측: 뇌졸중",
            "행 합계",
        ],
    )


# =====================================
# 8. 혼동행렬 세 개를 나란히 표시
# =====================================
st.subheader("3. 모델별 혼동행렬")

st.caption(
    "행은 실제 결과, 열은 예측 결과입니다. "
    "TP는 뇌졸중을 뇌졸중으로 맞힌 사람, "
    "TN은 뇌졸중 아님을 맞힌 사람, "
    "FP는 아닌데 뇌졸중이라고 예측한 사람, "
    "FN은 뇌졸중인데 놓친 사람입니다."
)

matrix_columns = st.columns(3)

for col, (model_name, predictions) in zip(
    matrix_columns,
    predictions_by_model.items(),
):
    with col:
        st.markdown(f"**{model_name}**")
        st.dataframe(
            make_confusion_table(y_test, predictions),
            use_container_width=True,
        )


# =====================================
# 9. 네 가지 지표 계산
# =====================================
def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = get_counts(y_true, y_pred)

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0

    # 정밀도: 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율
    predicted_positive = tp + fp
    if predicted_positive == 0:
        precision = "계산할 수 없음"
    else:
        precision = f"{tp / predicted_positive * 100:.1f}%"

    # 재현율: 실제 뇌졸중인 사람 중 맞게 찾아낸 비율
    actual_positive = tp + fn
    recall = tp / actual_positive if actual_positive else 0

    # 요청한 F1 공식
    f1_denominator = 2 * tp + fp + fn
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0
        else 0
    )

    return {
        "정확도": f"{accuracy * 100:.1f}%",
        "정밀도": precision,
        "재현율": f"{recall * 100:.1f}%",
        "F1": f"{f1 * 100:.1f}%",
    }


# =====================================
# 10. 지표 비교표
# =====================================
st.divider()
st.subheader("4. 세 모델의 평가지표 비교")

metrics_rows = []

for model_name, predictions in predictions_by_model.items():
    row = calculate_metrics(y_test, predictions)
    row["모델"] = model_name
    metrics_rows.append(row)

metrics_table = pd.DataFrame(metrics_rows)[
    ["모델", "정확도", "정밀도", "재현율", "F1"]
]

st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "정확도 = (TP + TN) / 전체 인원, "
    "정밀도 = TP / (TP + FP), "
    "재현율 = TP / (TP + FN), "
    "F1 = 2TP / (2TP + FP + FN)입니다. "
    "F1의 분모가 0이면 0.0%로 표시합니다."
)


# =====================================
# 11. 놓친 환자와 헛짚은 사람 목록
# =====================================
st.divider()
st.subheader("5. 모델의 예측 오류 확인")

chosen_model = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=list(predictions_by_model.keys()),
    key="score_error_model",
)

chosen_predictions = predictions_by_model[chosen_model]

# 실제 1, 예측 0 = FN
missed_mask = (y_test.to_numpy() == 1) & (chosen_predictions == 0)

# 실제 0, 예측 1 = FP
false_alarm_mask = (y_test.to_numpy() == 0) & (chosen_predictions == 1)

missed_df = test_df.loc[missed_mask].copy()
false_alarm_df = test_df.loc[false_alarm_mask].copy()


def make_people_list(source_df):
    """요청한 열만 선택하고 나이 많은 순으로 정렬한다."""
    if source_df.empty:
        return pd.DataFrame(
            columns=[
                "번호",
                "나이",
                "평균 혈당",
                "고혈압 여부",
                "심장병 여부",
            ]
        )

    if "id" in source_df.columns:
        numbers = source_df["id"].tolist()
    else:
        numbers = source_df.index.tolist()

    result = pd.DataFrame({
        "번호": numbers,
        "나이": pd.to_numeric(source_df["age"], errors="coerce").to_numpy(),
        "평균 혈당": pd.to_numeric(
            source_df["avg_glucose_level"], errors="coerce"
        ).to_numpy(),
        "고혈압 여부": pd.to_numeric(
            source_df["hypertension"], errors="coerce"
        ).map({0: "아니오", 1: "예"}).to_numpy(),
        "심장병 여부": pd.to_numeric(
            source_df["heart_disease"], errors="coerce"
        ).map({0: "아니오", 1: "예"}).to_numpy(),
    })

    return result.sort_values(
        "나이",
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


missed_list = make_people_list(missed_df)
false_alarm_list = make_people_list(false_alarm_df)

left, right = st.columns(2)

with left:
    st.markdown(
        f"**놓친 환자 — FN ({len(missed_list)}명)**"
    )
    st.caption("실제 뇌졸중이지만 모델은 뇌졸중이 아니라고 예측")

    if missed_list.empty:
        st.success("놓친 환자가 없습니다.")
    else:
        st.dataframe(
            missed_list.head(20),
            use_container_width=True,
            hide_index=True,
        )
        if len(missed_list) > 20:
            st.caption(
                f"전체 {len(missed_list)}명 중 "
                "나이가 많은 순으로 앞의 20명만 표시했습니다."
            )

with right:
    st.markdown(
        f"**헛짚은 사람 — FP ({len(false_alarm_list)}명)**"
    )
    st.caption("실제 뇌졸중은 아니지만 모델은 뇌졸중이라고 예측")

    if false_alarm_list.empty:
        st.success("헛짚은 사람이 없습니다.")
    else:
        st.dataframe(
            false_alarm_list.head(20),
            use_container_width=True,
            hide_index=True,
        )
        if len(false_alarm_list) > 20:
            st.caption(
                f"전체 {len(false_alarm_list)}명 중 "
                "나이가 많은 순으로 앞의 20명만 표시했습니다."
            )

st.info(
    f"목록의 사람 수는 혼동행렬과 일치합니다. "
    f"FN(놓친 환자) = {len(missed_list)}명, "
    f"FP(헛짚은 사람) = {len(false_alarm_list)}명."
)
