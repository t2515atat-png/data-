
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


st.title("🧹 전처리와 클래스 무게 균형 비교")
st.write(
    "같은 채점용 데이터를 사용해 전처리 및 클래스 무게 균형 적용 전후를 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RANDOM_STATE = 42

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


@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


# --------------------------------------------------
# 1. 화면 맨 위의 세 가지 설정
# --------------------------------------------------
st.subheader("1. 분석 설정")

balance_enabled = st.toggle(
    "클래스 무게 균형 적용",
    value=False,
    help=(
        "켜면 로지스틱 회귀와 결정 트리 모두 "
        "class_weight='balanced'로 학습합니다."
    ),
)

bmi_action = st.radio(
    "BMI가 비어 있는 사람 처리",
    options=["그대로 두기", "삭제하기"],
    horizontal=True,
)

selected_features = st.multiselect(
    "분류 모델의 입력 항목",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x],
    key="preprocess_features",
)

if len(selected_features) < 2:
    st.warning("입력 항목을 두 개 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 2. 데이터 불러오기 및 채점표와 같은 순서로 분할
# --------------------------------------------------
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
    st.error("훈련용과 채점용 데이터를 나눌 수 있는 인원이 부족합니다.")
    st.stop()

# 채점표와 동일하게 정렬한 뒤 앞의 3명을 채점용으로 고정
test_original = df.iloc[:3].copy()
train_original = df.iloc[3:].copy()


# --------------------------------------------------
# 3. BMI 결측값 처리
# --------------------------------------------------
bmi_missing_all = df["bmi"].isna()
deleted_count = 0
deleted_stroke_count = 0

if bmi_action == "삭제하기":
    deleted_count = int(bmi_missing_all.sum())
    deleted_stroke_count = int(
        (bmi_missing_all & (df["stroke"] == 1)).sum()
    )

    # 훈련용과 채점용의 원래 구분은 유지하고,
    # BMI가 비어 있는 행만 각 집합에서 제거
    train_df = train_original[train_original["bmi"].notna()].copy()
    test_df = test_original[test_original["bmi"].notna()].copy()

    st.info(
        f"BMI 결측으로 삭제한 사람: {deleted_count}명 "
        f"(그중 실제 뇌졸중 환자 {deleted_stroke_count}명)"
    )
else:
    train_df = train_original.copy()
    test_df = test_original.copy()
    st.info(
        f"BMI 결측값을 그대로 두었습니다. "
        f"전체 BMI 결측 인원은 {int(bmi_missing_all.sum())}명입니다."
    )

if len(train_df) == 0:
    st.error("BMI 처리 후 훈련용 데이터가 남아 있지 않습니다.")
    st.stop()

if len(test_df) == 0:
    st.error(
        "채점용으로 고정한 앞의 3명이 모두 제거되었습니다. "
        "BMI 삭제 대신 '그대로 두기'를 선택해 주세요."
    )
    st.stop()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error(
        "훈련용 데이터에 뇌졸중 경험자와 비경험자가 모두 있어야 "
        "모델을 학습할 수 있습니다."
    )
    st.stop()

st.caption(
    f"훈련용 데이터: {len(train_df):,}명 / "
    f"채점용 데이터: {len(test_df):,}명"
)


# --------------------------------------------------
# 4. 입력값 준비
# --------------------------------------------------
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

for col in selected_features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # BMI를 삭제하지 않는 경우에는 훈련용 중앙값으로 채운다.
    # 다른 선택 항목에 결측값이 있더라도 모델 오류를 막기 위해
    # 해당 항목의 훈련용 중앙값을 사용한다.
    if X_train[col].isna().any():
        median_value = X_train[col].median()

        if pd.isna(median_value):
            st.error(
                f"'{FEATURE_LABELS[col]}' 항목의 훈련용 값이 모두 비어 있습니다."
            )
            st.stop()

        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)

# 스케일링은 로지스틱 회귀에만 사용한다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 5. 모델 학습: 무게 균형 적용 전후를 항상 계산
# --------------------------------------------------
def train_models(class_weight):
    logistic = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=2000,
        class_weight=class_weight,
    )
    logistic.fit(X_train_scaled, y_train)

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    tree.fit(X_train, y_train)

    return logistic, tree


try:
    logistic_before, tree_before = train_models(None)
    logistic_after, tree_after = train_models("balanced")
except Exception as e:
    st.error("모델 학습 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


def positive_probability(model, X):
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, positive_index]


prob_before = positive_probability(logistic_before, X_test_scaled)
prob_after = positive_probability(logistic_after, X_test_scaled)

predictions = {
    "확률 모델 · 전": (prob_before >= 0.5).astype(int),
    "확률 모델 · 후": (prob_after >= 0.5).astype(int),
    "질문 모델 · 전": tree_before.predict(X_test).astype(int),
    "질문 모델 · 후": tree_after.predict(X_test).astype(int),
}


# --------------------------------------------------
# 6. 혼동행렬과 평가지표
# --------------------------------------------------
def get_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    return tn, fp, fn, tp


def confusion_table(y_true, y_pred):
    tn, fp, fn, tp = get_counts(y_true, y_pred)

    return pd.DataFrame(
        [
            [f"TN {tn}명", f"FP {fp}명"],
            [f"FN {fn}명", f"TP {tp}명"],
        ],
        index=["실제: 뇌졸중 아님", "실제: 뇌졸중"],
        columns=["예측: 뇌졸중 아님", "예측: 뇌졸중"],
    )


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = get_counts(y_true, y_pred)
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 0

    return {
        "정확도": f"{accuracy * 100:.1f}%",
        "정밀도": f"{precision * 100:.1f}%",
        "재현율": f"{recall * 100:.1f}%",
        "F1": f"{f1 * 100:.1f}%",
        "찾아낸 환자": tp,
        "놓친 환자": fn,
        "헛짚은 사람": fp,
    }


st.divider()
st.subheader("2. 모델별 혼동행렬과 평가지표")

for model_title, before_key, after_key in [
    ("확률 모델 · 로지스틱 회귀", "확률 모델 · 전", "확률 모델 · 후"),
    ("질문 모델 · 결정 트리", "질문 모델 · 전", "질문 모델 · 후"),
]:
    st.markdown(f"**{model_title}**")
    left, right = st.columns(2)

    with left:
        st.caption("무게 균형 적용 전")
        st.dataframe(
            confusion_table(y_test, predictions[before_key]),
            use_container_width=True,
        )
        before_metrics = calculate_metrics(y_test, predictions[before_key])
        st.dataframe(
            pd.DataFrame([before_metrics]).drop(
                columns=["찾아낸 환자", "놓친 환자", "헛짚은 사람"]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.caption("무게 균형 적용 후")
        st.dataframe(
            confusion_table(y_test, predictions[after_key]),
            use_container_width=True,
        )
        after_metrics = calculate_metrics(y_test, predictions[after_key])
        st.dataframe(
            pd.DataFrame([after_metrics]).drop(
                columns=["찾아낸 환자", "놓친 환자", "헛짚은 사람"]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        f"찾아낸 환자: {before_metrics['찾아낸 환자']}명 → "
        f"{after_metrics['찾아낸 환자']}명 | "
        f"놓친 환자: {before_metrics['놓친 환자']}명 → "
        f"{after_metrics['놓친 환자']}명 | "
        f"헛짚은 사람: {before_metrics['헛짚은 사람']}명 → "
        f"{after_metrics['헛짚은 사람']}명"
    )

st.caption(
    "TN은 뇌졸중이 아님을 맞힌 사람, FP는 뇌졸중이라고 잘못 예측한 사람, "
    "FN은 실제 환자를 놓친 사람, TP는 실제 환자를 찾아낸 사람입니다."
)


# --------------------------------------------------
# 7. 산점도: 두 항목은 2차원, 세 항목 이상은 3차원
# --------------------------------------------------
st.divider()
st.subheader("3. 입력 항목과 실제 결과 및 0.5 경계")

plot_df = X_test.copy()
plot_df["실제 결과"] = y_test.map({0: "뇌졸중 아님", 1: "뇌졸중 환자"}).to_numpy()
plot_df["실제 결과"] = plot_df["실제 결과"].astype(str)

# 시각화는 선택한 항목 중 앞의 2개 또는 3개를 축으로 사용한다.
if len(selected_features) == 2:
    x_feature, y_feature = selected_features
    st.caption(
        f"가로축: {FEATURE_LABELS[x_feature]} / "
        f"세로축: {FEATURE_LABELS[y_feature]}. "
        "경계선은 확률 모델의 예측 확률 0.5를 나타냅니다."
    )

    fig = px.scatter(
        plot_df,
        x=x_feature,
        y=y_feature,
        color="실제 결과",
        hover_data=selected_features,
        labels={
            x_feature: FEATURE_LABELS[x_feature],
            y_feature: FEATURE_LABELS[y_feature],
            "실제 결과": "실제 뇌졸중 여부",
        },
        title="채점용 데이터와 확률 모델의 0.5 경계선",
    )

    x_min, x_max = float(X_test[x_feature].min()), float(X_test[x_feature].max())
    y_min, y_max = float(X_test[y_feature].min()), float(X_test[y_feature].max())

    # 채점용 데이터의 다른 입력 항목은 중앙값으로 고정한다.
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )

    grid = pd.DataFrame(
        np.tile(X_train.median().to_numpy(), (grid_x.size, 1)),
        columns=selected_features,
    )
    grid[x_feature] = grid_x.ravel()
    grid[y_feature] = grid_y.ravel()
    grid_scaled = scaler.transform(grid)

    z_before = positive_probability(
        logistic_before, grid_scaled
    ).reshape(grid_x.shape)
    z_after = positive_probability(
        logistic_after, grid_scaled
    ).reshape(grid_x.shape)

    # 0.5 등고선은 해당 경계가 축 범위 안에 있을 때만 그려진다.
    before_has_boundary = z_before.min() <= 0.5 <= z_before.max()
    after_has_boundary = z_after.min() <= 0.5 <= z_after.max()

    if before_has_boundary:
        fig.add_trace(
            go.Contour(
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                z=z_before,
                contours=dict(start=0.5, end=0.5, size=1),
                showscale=False,
                name="무게 균형 전 · 0.5",
                line=dict(color="blue", width=3),
                hoverinfo="skip",
            )
        )

    if after_has_boundary:
        fig.add_trace(
            go.Contour(
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                z=z_after,
                contours=dict(start=0.5, end=0.5, size=1),
                showscale=False,
                name="무게 균형 후 · 0.5",
                line=dict(color="red", width=3, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(legend_title_text="실제 결과 / 경계선")
    st.plotly_chart(fig, use_container_width=True)

    if not before_has_boundary:
        st.write("무게 균형 전 0.5 경계선은 현재 그림 범위 밖에 있습니다.")
    if not after_has_boundary:
        st.write("무게 균형 후 0.5 경계선은 현재 그림 범위 밖에 있습니다.")

else:
    x_feature, y_feature, z_feature = selected_features[:3]
    st.caption(
        f"3차원 축: {FEATURE_LABELS[x_feature]}, "
        f"{FEATURE_LABELS[y_feature]}, {FEATURE_LABELS[z_feature]}. "
        "마우스로 드래그해 회전할 수 있습니다."
    )

    fig = px.scatter_3d(
        plot_df,
        x=x_feature,
        y=y_feature,
        z=z_feature,
        color="실제 결과",
        hover_data=selected_features,
        labels={
            x_feature: FEATURE_LABELS[x_feature],
            y_feature: FEATURE_LABELS[y_feature],
            z_feature: FEATURE_LABELS[z_feature],
            "실제 결과": "실제 뇌졸중 여부",
        },
        title="3차원 산점도와 확률 모델의 0.5 평면",
    )

    # 3차원 경계면을 만들기 위해 나머지 입력값은 훈련용 중앙값으로 고정한다.
    x_min, x_max = float(X_test[x_feature].min()), float(X_test[x_feature].max())
    y_min, y_max = float(X_test[y_feature].min()), float(X_test[y_feature].max())
    z_min, z_max = float(X_test[z_feature].min()), float(X_test[z_feature].max())

    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, 35),
        np.linspace(y_min, y_max, 35),
    )

    def calculate_boundary_surface(model):
        coef = model.coef_[0]
        intercept = float(model.intercept_[0])

        means = scaler.mean_
        scales = scaler.scale_

        # 원래 단위의 입력값으로 바꾼 로지스틱 회귀 방정식
        raw_coef = coef / scales
        raw_intercept = intercept - float(np.sum(coef * means / scales))

        fixed_values = X_train.median().to_dict()
        fixed_sum = raw_intercept

        for feature in selected_features:
            if feature not in [x_feature, y_feature, z_feature]:
                fixed_sum += raw_coef[selected_features.index(feature)] * float(
                    fixed_values[feature]
                )

        ix = selected_features.index(x_feature)
        iy = selected_features.index(y_feature)
        iz = selected_features.index(z_feature)

        denominator = raw_coef[iz]
        if abs(denominator) < 1e-10:
            return None

        gz = -(
            fixed_sum + raw_coef[ix] * gx + raw_coef[iy] * gy
        ) / denominator

        if np.nanmax(gz) < z_min or np.nanmin(gz) > z_max:
            return None

        gz = np.where((gz >= z_min) & (gz <= z_max), gz, np.nan)
        return gz

    surface_before = calculate_boundary_surface(logistic_before)
    surface_after = calculate_boundary_surface(logistic_after)

    if surface_before is not None:
        fig.add_trace(
            go.Surface(
                x=gx,
                y=gy,
                z=surface_before,
                name="무게 균형 전 · 0.5 평면",
                opacity=0.45,
                showscale=False,
                colorscale=[[0, "blue"], [1, "blue"]],
                hoverinfo="skip",
            )
        )

    if surface_after is not None:
        fig.add_trace(
            go.Surface(
                x=gx,
                y=gy,
                z=surface_after,
                name="무게 균형 후 · 0.5 평면",
                opacity=0.45,
                showscale=False,
                colorscale=[[0, "red"], [1, "red"]],
                hoverinfo="skip",
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    if surface_before is None:
        st.write(
            "무게 균형 전 0.5 평면은 현재 3차원 그림 범위 안에 표시되지 않습니다."
        )
    if surface_after is None:
        st.write(
            "무게 균형 후 0.5 평면은 현재 3차원 그림 범위 안에 표시되지 않습니다."
        )

    if len(selected_features) > 3:
        st.caption(
            "입력 항목이 4개 이상이므로 그림에는 앞의 3개 항목을 축으로 사용했습니다. "
            "나머지 항목은 훈련용 중앙값으로 고정해 경계면을 계산했습니다."
        )


# --------------------------------------------------
# 8. 확률 모델의 놓친 환자 목록
# --------------------------------------------------
st.divider()
st.subheader("4. 확률 모델이 놓친 환자")

if balance_enabled:
    missed_predictions = predictions["확률 모델 · 후"]
    list_title = "클래스 무게 균형 적용 후"
else:
    missed_predictions = predictions["확률 모델 · 전"]
    list_title = "클래스 무게 균형 적용 전"

missed_mask = (y_test.to_numpy() == 1) & (missed_predictions == 0)
missed_people = test_df.loc[missed_mask].copy()

st.caption(f"현재 스위치 기준: {list_title}")

if missed_people.empty:
    st.success("놓친 환자가 없습니다.")
else:
    if "id" in missed_people.columns:
        person_numbers = missed_people["id"].to_numpy()
    else:
        person_numbers = missed_people.index.to_numpy()

    missed_list = pd.DataFrame({
        "번호": person_numbers,
        "나이": pd.to_numeric(
            missed_people["age"], errors="coerce"
        ).to_numpy(),
        "고혈압 여부": pd.to_numeric(
            missed_people["hypertension"], errors="coerce"
        ).map({0: "아니오", 1: "예"}).to_numpy(),
        "심장병 여부": pd.to_numeric(
            missed_people["heart_disease"], errors="coerce"
        ).map({0: "아니오", 1: "예"}).to_numpy(),
    })

    missed_list = missed_list.sort_values(
        "나이",
        ascending=True,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    st.dataframe(
        missed_list,
        use_container_width=True,
        hide_index=True,
    )

st.info(
    "이 페이지는 학습용 데이터 분석 실습입니다. "
    "실제 의료 진단이나 치료 판단에 사용해서는 안 됩니다."
)
