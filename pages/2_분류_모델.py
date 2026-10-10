
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score


# =====================================
# 1. 기본 설정
# =====================================
st.title("🌳 뇌졸중 분류 모델")
st.write(
    "입력 속성을 선택하고, 확률 모델과 질문 모델의 "
    "예측 정확도를 비교합니다."
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
# 2. 데이터 불러오기
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

missing_columns = [
    c for c in required_columns if c not in df.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

# 번호 순으로 정렬. id가 없으면 원래 행 순서를 사용한다.
if "id" in df.columns:
    df = df.sort_values("id", kind="stable").reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# 목표값은 0 또는 1인 행만 사용한다.
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")
df = df[df["stroke"].isin([0, 1])].copy()
df["stroke"] = df["stroke"].astype(int)

# 앞의 3명을 채점용으로 고정하고 나머지는 훈련용으로 사용한다.
if len(df) < 4:
    st.error("분류 모델을 학습하려면 유효한 데이터가 최소 4명 필요합니다.")
    st.stop()

test_df = df.iloc[:3].copy()
train_df = df.iloc[3:].copy()

st.caption(
    f"전체 {len(df):,}명 중 앞의 {len(test_df)}명은 채점용, "
    f"나머지 {len(train_df):,}명은 훈련용입니다."
)


# =====================================
# 3. 입력 속성 선택
# =====================================
st.subheader("1. 입력 속성 선택")

selected_features = st.multiselect(
    "모델에 사용할 속성을 선택하세요.",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x],
    key="classification_features",
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# 선택한 속성만 숫자로 변환
for col in selected_features:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error(
        "훈련용 데이터에 뇌졸중 경험자와 비경험자가 모두 있어야 "
        "분류 모델을 학습할 수 있습니다."
    )
    st.stop()


# =====================================
# 4. 결측값 처리 및 크기 맞추기
# =====================================
# BMI가 선택된 경우에만 훈련용 BMI 중앙값으로 채운다.
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련용 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 선택한 속성 중 다른 값에 결측치가 있으면
# 훈련용 중앙값으로 처리한다.
for col in selected_features:
    if X_train[col].isna().any():
        median_value = X_train[col].median()

        if pd.isna(median_value):
            st.error(f"'{FEATURE_LABELS[col]}'의 훈련용 값이 모두 비어 있습니다.")
            st.stop()

        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)

# 훈련용 데이터로만 크기 맞추기 기준을 학습한다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================
# 5. 확률 모델: 로지스틱 회귀
# =====================================
probability_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=2000,
)

probability_model.fit(X_train_scaled, y_train)

probabilities = probability_model.predict_proba(X_test_scaled)
positive_index = list(probability_model.classes_).index(1)
probability_predictions = (
    probabilities[:, positive_index] >= 0.5
).astype(int)


# =====================================
# 6. 질문 모델: 결정 트리
# =====================================
# 결정 트리는 변수 크기에 영향을 받지 않으므로
# 원래 단위의 값을 사용한다.
question_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)

question_model.fit(X_train, y_train)
question_predictions = question_model.predict(X_test).astype(int)


# =====================================
# 7. 아무 입력도 보지 않는 기준 모델
# =====================================
# 훈련용에서 더 많은 쪽의 값만 예측한다.
majority_class = int(y_train.value_counts().idxmax())
baseline_predictions = np.full(len(y_test), majority_class)

probability_accuracy = accuracy_score(
    y_test, probability_predictions
)
question_accuracy = accuracy_score(
    y_test, question_predictions
)
baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)


# =====================================
# 8. 정확도 카드
# =====================================
st.divider()
st.subheader("2. 채점용 데이터 예측 정확도")

c1, c2, c3 = st.columns(3)

c1.metric(
    "확률 모델",
    f"{probability_accuracy * 100:.1f}%",
)

c2.metric(
    "질문 모델",
    f"{question_accuracy * 100:.1f}%",
)

c3.metric(
    "다수 집단만 예측",
    f"{baseline_accuracy * 100:.1f}%",
)

st.caption(
    "채점용 데이터는 앞의 3명뿐이므로 정확도는 0%, 33.3%, "
    "66.7%, 100% 중 하나가 될 수 있습니다. "
    "채점용 데이터는 학습이나 크기 맞추기에 사용하지 않았습니다."
)


# =====================================
# 9. 예측 결과 표
# =====================================
st.subheader("3. 채점용 사람별 예측 결과")

result_table = pd.DataFrame({
    "데이터 번호": (
        test_df["id"].tolist()
        if "id" in test_df.columns
        else test_df.index.tolist()
    ),
    "실제 결과": [
        "뇌졸중" if x == 1 else "뇌졸중 아님"
        for x in y_test
    ],
    "확률 모델 예측": [
        "뇌졸중" if x == 1 else "뇌졸중 아님"
        for x in probability_predictions
    ],
    "뇌졸중 확률(%)": np.round(
        probabilities[:, positive_index] * 100, 1
    ),
    "질문 모델 예측": [
        "뇌졸중" if x == 1 else "뇌졸중 아님"
        for x in question_predictions
    ],
    "다수 집단 모델 예측": [
        "뇌졸중" if x == 1 else "뇌졸중 아님"
        for x in baseline_predictions
    ],
})

st.dataframe(
    result_table,
    use_container_width=True,
    hide_index=True,
)


# =====================================
# 10. 질문 모델의 질문 순서
# =====================================
st.divider()
st.subheader("4. 질문 모델이 던지는 질문 순서")

tree = question_model.tree_
feature_names = list(selected_features)


def describe_tree(node_id=0, depth=0):
    """결정 트리의 질문을 왼쪽 가지부터 순서대로 표시한다."""
    indent = "　" * depth

    if tree.feature[node_id] == _tree.TREE_UNDEFINED:
        counts = tree.value[node_id][0]
        total = int(counts.sum())

        if total == 0:
            prediction = "자료 없음"
        else:
            prediction = (
                "뇌졸중"
                if question_model.classes_[int(np.argmax(counts))] == 1
                else "뇌졸중 아님"
            )

        st.write(
            f"{indent}→ 마지막 마디: {prediction} "
            f"(훈련용 {total}명)"
        )
        return

    feature_index = tree.feature[node_id]
    feature = feature_names[feature_index]
    threshold = tree.threshold[node_id]
    label = FEATURE_LABELS[feature]

    if feature in ["hypertension", "heart_disease"]:
        st.write(
            f"{indent}질문: {label} 값이 "
            f"{threshold:.2f} 이하인가?"
        )
    else:
        st.write(
            f"{indent}질문: {label}이 "
            f"{threshold:.2f} 이하인가?"
        )

    describe_tree(tree.children_left[node_id], depth + 1)
    describe_tree(tree.children_right[node_id], depth + 1)


describe_tree()


# =====================================
# 11. 2개 속성 선택 및 시각화
# =====================================
st.divider()
st.subheader("5. 두 속성으로 모델 경계 살펴보기")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="plot_x_feature",
    )

with axis_col2:
    y_options = [x for x in selected_features if x != x_feature]
    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="plot_y_feature",
    )

# 시각화에 사용할 두 속성만으로 별도 모델 학습
X2_train = train_df[[x_feature, y_feature]].copy()
X2_test = test_df[[x_feature, y_feature]].copy()

# 두 축 중 BMI가 선택되었을 때만 훈련용 중앙값으로 채운다.
for feature in [x_feature, y_feature]:
    X2_train[feature] = pd.to_numeric(
        X2_train[feature], errors="coerce"
    )
    X2_test[feature] = pd.to_numeric(
        X2_test[feature], errors="coerce"
    )

    if feature == "bmi":
        median_value = X2_train[feature].median()
        if pd.isna(median_value):
            st.error("두 축에 사용할 BMI 중앙값을 계산할 수 없습니다.")
            st.stop()

        X2_train[feature] = X2_train[feature].fillna(median_value)
        X2_test[feature] = X2_test[feature].fillna(median_value)

    if X2_train[feature].isna().any():
        median_value = X2_train[feature].median()

        if pd.isna(median_value):
            st.error(
                f"'{FEATURE_LABELS[feature]}'의 값이 없어 "
                "그래프를 만들 수 없습니다."
            )
            st.stop()

        X2_train[feature] = X2_train[feature].fillna(median_value)
        X2_test[feature] = X2_test[feature].fillna(median_value)

# 2차원 확률 모델도 훈련용만으로 크기 맞추기
scaler_2d = StandardScaler()
X2_train_scaled = scaler_2d.fit_transform(X2_train)

probability_model_2d = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=2000,
)
probability_model_2d.fit(X2_train_scaled, y_train)

# 두 축으로 만든 질문 모델
question_model_2d = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
question_model_2d.fit(X2_train, y_train)


# =====================================
# 12. 그래프 영역 만들기
# =====================================
x_values = X2_train[x_feature].to_numpy()
y_values = X2_train[y_feature].to_numpy()

x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
y_min, y_max = float(np.min(y_values)), float(np.max(y_values))

x_padding = (x_max - x_min) * 0.08
y_padding = (y_max - y_min) * 0.08

if x_padding == 0:
    x_padding = 0.5
if y_padding == 0:
    y_padding = 0.5

x_grid = np.linspace(x_min - x_padding, x_max + x_padding, 180)
y_grid = np.linspace(y_min - y_padding, y_max + y_padding, 180)

xx, yy = np.meshgrid(x_grid, y_grid)
grid_2d = pd.DataFrame({
    x_feature: xx.ravel(),
    y_feature: yy.ravel(),
})

# 질문 모델이 나누는 칸
tree_region = question_model_2d.predict(grid_2d).reshape(xx.shape)

# 확률 모델의 뇌졸중 확률
grid_scaled = scaler_2d.transform(grid_2d)
grid_probability = probability_model_2d.predict_proba(grid_scaled)
positive_2d_index = list(probability_model_2d.classes_).index(1)
probability_region = grid_probability[:, positive_2d_index].reshape(xx.shape)

# 0.5 경계가 그림 범위 안에 있는지 확인
prob_min = float(np.min(probability_region))
prob_max = float(np.max(probability_region))
boundary_inside = prob_min <= 0.5 <= prob_max


# =====================================
# 13. Plotly 그래프
# =====================================
fig = go.Figure()

# 질문 모델의 분류 영역을 옅은 색으로 표시
fig.add_trace(
    go.Heatmap(
        x=x_grid,
        y=y_grid,
        z=tree_region,
        zmin=0,
        zmax=1,
        colorscale=[
            [0.0, "rgba(110,160,230,0.15)"],
            [0.499, "rgba(110,160,230,0.15)"],
            [0.5, "rgba(240,130,130,0.15)"],
            [1.0, "rgba(240,130,130,0.15)"],
        ],
        showscale=False,
        hoverinfo="skip",
        name="질문 모델 영역",
    )
)

# 확률 모델의 0.5 경계선
if boundary_inside:
    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=probability_region,
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 1,
                "coloring": "none",
                "showlabels": False,
            },
            line={
                "color": "black",
                "width": 3,
            },
            showscale=False,
            hoverinfo="skip",
            name="확률 0.5 경계",
        )
    )

# 채점용 사람은 실제 뇌졸중 여부에 따라 색칠
test_x = X2_test[x_feature].to_numpy()
test_y = X2_test[y_feature].to_numpy()
actual_y = y_test.to_numpy()

for actual_value, label, color in [
    (0, "실제 뇌졸중 아님", "#2878B5"),
    (1, "실제 뇌졸중", "#D62728"),
]:
    mask = actual_y == actual_value

    fig.add_trace(
        go.Scatter(
            x=test_x[mask],
            y=test_y[mask],
            mode="markers",
            name=label,
            marker={
                "size": 13,
                "color": color,
                "line": {"color": "white", "width": 1.5},
            },
            text=[
                f"실제 결과: {label}"
                for _ in range(int(mask.sum()))
            ],
            hovertemplate=(
                f"{FEATURE_LABELS[x_feature]}: %{{x:.2f}}<br>"
                f"{FEATURE_LABELS[y_feature]}: %{{y:.2f}}<br>"
                "%{text}<extra></extra>"
            ),
        )
    )

fig.update_layout(
    title="채점용 데이터와 두 모델의 분류 경계",
    xaxis_title=FEATURE_LABELS[x_feature],
    yaxis_title=FEATURE_LABELS[y_feature],
    legend_title="실제 결과",
    height=650,
)

st.plotly_chart(fig, use_container_width=True)

if boundary_inside:
    st.success(
        "확률 모델의 뇌졸중 확률 0.5 경계선이 "
        "현재 그림 범위 안에 있습니다."
    )
else:
    st.warning(
        "확률 모델의 뇌졸중 확률 0.5 경계선이 "
        "현재 그림 범위 밖에 있어 표시되지 않습니다."
    )

st.caption(
    "옅은 파란색과 붉은색 영역은 두 속성만 사용한 질문 모델의 "
    "분류 영역입니다. 검은 선은 두 속성만 사용한 확률 모델의 "
    "뇌졸중 확률 0.5 경계입니다. 점의 색은 실제 결과를 뜻합니다."
)
