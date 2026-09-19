import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 흥행 예측기")
st.write(
    "영화 정보 데이터를 이용하여 영화의 총 관객 수를 "
    "다중 회귀 모델로 예측합니다."
)


# =========================================================
# 데이터 주소
# =========================================================
DAILY_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/kobis_daily.csv"
)

MOVIES_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/kobis_movies.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    daily = pd.read_csv(
        DAILY_URL,
        encoding="utf-8"
    )

    movies = pd.read_csv(
        MOVIES_URL,
        encoding="utf-8"
    )

    return daily, movies


try:
    daily, movies = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# =========================================================
# 일별 데이터 기간
# =========================================================
daily["날짜"] = pd.to_datetime(
    daily["날짜"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

date_min = daily["날짜"].min()
date_max = daily["날짜"].max()


# =========================================================
# 영화코드 정리
# =========================================================
movies["movieCd"] = movies["movieCd"].astype(str).str.strip()

# 영화코드 순으로 정렬
movies = movies.sort_values(
    "movieCd",
    key=lambda x: pd.to_numeric(x, errors="coerce")
).reset_index(drop=True)


# =========================================================
# 영화 정보 표 맨 위 행 표시
# =========================================================
st.subheader("📋 영화 정보 표")

st.write(
    f"기준 기간: **{date_min.strftime('%Y-%m-%d')} ~ "
    f"{date_max.strftime('%Y-%m-%d')}**"
)

st.write("영화 정보 표의 맨 위 행:")

st.dataframe(
    movies.head(1),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 사용할 수 있는 변수
# =========================================================
st.subheader("⚙️ 예측에 사용할 변수 선택")

st.write(
    "체크한 변수만 다중 회귀 모델의 입력값으로 사용됩니다. "
    "`total_audi`는 예측 대상이므로 입력 변수에서 제외됩니다."
)


# 변수 설명
variable_info = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권에 머문 일수",
    "genre": "장르",
    "nation": "국가",
    "openDt": "개봉일"
}

available_variables = [
    col for col in variable_info
    if col in movies.columns
]


# 기본 선택 변수
default_variables = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "days_in_top10",
    "genre",
    "nation"
]

selected_variables = []

cols = st.columns(3)

for i, variable in enumerate(available_variables):

    with cols[i % 3]:

        checked = st.checkbox(
            f"{variable_info[variable]} ({variable})",
            value=variable in default_variables
        )

        if checked:
            selected_variables.append(variable)


if len(selected_variables) == 0:
    st.warning("최소 한 개의 변수를 선택해주세요.")
    st.stop()


st.write(
    "**현재 선택된 변수:** "
    + ", ".join(selected_variables)
)


# =========================================================
# 데이터 준비
# =========================================================
model_df = movies.copy()

target = "total_audi"

# 필요한 열만 사용
needed_columns = selected_variables + [target, "movieCd", "movieNm"]

model_df = model_df[needed_columns].copy()


# 숫자형 변수 변환
numeric_candidates = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "days_in_top10"
]

for col in numeric_candidates:
    if col in model_df.columns:
        model_df[col] = pd.to_numeric(
            model_df[col],
            errors="coerce"
        )


# 개봉일에서 연도/월/일 정보를 숫자로 변환
# 문자열 그대로 넣는 대신 숫자형 변수로 변환
if "openDt" in model_df.columns:

    open_date = pd.to_datetime(
        model_df["openDt"],
        errors="coerce"
    )

    model_df["open_year"] = open_date.dt.year
    model_df["open_month"] = open_date.dt.month
    model_df["open_day"] = open_date.dt.day

    model_df = model_df.drop(columns=["openDt"])

    selected_variables = [
        "open_year",
        "open_month",
        "open_day"
        if "open_day" in model_df.columns
        else "open_month"
    ] + [
        x for x in selected_variables
        if x != "openDt"
    ]


# target 숫자 변환
model_df[target] = pd.to_numeric(
    model_df[target],
    errors="coerce"
)


# =========================================================
# 결측값 처리
# =========================================================
model_df = model_df.dropna(
    subset=[target]
).reset_index(drop=True)


# =========================================================
# 10편마다 앞의 3편을 테스트용으로 분리
# 나머지 7편은 학습
# =========================================================
train_indices = []
test_indices = []

for start in range(0, len(model_df), 10):

    block_indices = list(
        range(
            start,
            min(start + 10, len(model_df))
        )
    )

    # 각 10편 묶음의 앞 3편
    test_block = block_indices[:3]

    # 나머지
    train_block = block_indices[3:]

    test_indices.extend(test_block)
    train_indices.extend(train_block)


train_df = model_df.iloc[train_indices].copy()
test_df = model_df.iloc[test_indices].copy()


# =========================================================
# 입력 변수 / 정답
# =========================================================
X_train = train_df[selected_variables]
y_train = train_df[target]

X_test = test_df[selected_variables]
y_test = test_df[target]


# =========================================================
# 변수 종류 자동 구분
# =========================================================
categorical_features = [
    col for col in selected_variables
    if model_df[col].dtype == "object"
]

numeric_features = [
    col for col in selected_variables
    if col not in categorical_features
]


# =========================================================
# 전처리
# =========================================================
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


transformers = []

if numeric_features:
    transformers.append(
        (
            "numeric",
            numeric_transformer,
            numeric_features
        )
    )

if categorical_features:
    transformers.append(
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers
)


# =========================================================
# 다중 회귀 모델
# =========================================================
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regression",
            LinearRegression()
        )
    ]
)


# 학습
model.fit(
    X_train,
    y_train
)


# 테스트 영화 예측
y_pred = model.predict(X_test)


# =========================================================
# 예측 결과 정리
# =========================================================
result = test_df[
    ["movieCd", "movieNm", "total_audi"]
].copy()

result["예측 총 관객"] = y_pred
result["실제 총 관객"] = y_test.to_numpy()

result["오차"] = (
    result["예측 총 관객"]
    - result["실제 총 관객"]
)

result["절대오차"] = (
    result["오차"].abs()
)


# =========================================================
# 평가 점수
# =========================================================
r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


# =========================================================
# 화면 정보
# =========================================================
st.subheader("📊 모델 평가")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "학습에 사용한 영화",
        f"{len(train_df)}편"
    )

with col2:
    st.metric(
        "평가한 영화",
        f"{len(test_df)}편"
    )

with col3:
    st.metric(
        "R² 점수",
        f"{r2:.3f}"
    )

with col4:
    st.metric(
        "평균 절대 오차",
        f"{mae:,.0f}명"
    )


st.write(
    f"**기준 기간:** "
    f"{date_min.strftime('%Y-%m-%d')} ~ "
    f"{date_max.strftime('%Y-%m-%d')}"
)

st.write(
    f"**RMSE:** {rmse:,.0f}명"
)

st.caption(
    "R²는 테스트용 영화에서 실제 관객 수와 예측값이 "
    "얼마나 잘 맞는지를 나타내는 평가 점수입니다. "
    "평균 절대 오차는 예측이 실제 관객 수에서 평균적으로 "
    "몇 명 정도 벗어났는지를 나타냅니다."
)


# =========================================================
# 예측 결과 표
# =========================================================
st.subheader("🎞️ 테스트 영화의 실제값과 예측값")

display_result = result[
    [
        "movieCd",
        "movieNm",
        "실제 총 관객",
        "예측 총 관객",
        "오차",
        "절대오차"
    ]
].copy()

display_result["실제 총 관객"] = (
    display_result["실제 총 관객"].round(0).astype(int)
)

display_result["예측 총 관객"] = (
    display_result["예측 총 관객"].round(0)
)

display_result["오차"] = (
    display_result["오차"].round(0)
)

display_result["절대오차"] = (
    display_result["절대오차"].round(0)
)

st.dataframe(
    display_result,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 로그 산점도
# =========================================================
st.subheader("📈 실제 총 관객 수와 예측 총 관객 수")

# 로그축에서는 0 이하 값을 표시할 수 없음
# 실제 관객은 양수라고 가정하고, 예측값은 1보다 작으면 1로 표시
actual_plot = np.maximum(
    result["실제 총 관객"].to_numpy(),
    1
)

prediction_raw = result["예측 총 관객"].to_numpy()

# 1,000명 미만 예측값 개수
low_prediction_count = np.sum(
    prediction_raw < 1000
)

# 그래프에서 1000명 미만 예측값은 바닥에 표시
prediction_plot = np.maximum(
    prediction_raw,
    1
)

# 일반 예측값
normal_mask = prediction_raw >= 1000

# 1000명 미만 예측값
low_mask = prediction_raw < 1000


fig = go.Figure()


# ---------------------------------------------------------
# 일반 예측값
# ---------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=actual_plot[normal_mask],
        y=prediction_plot[normal_mask],
        mode="markers",
        name="예측값",
        customdata=result.loc[
            normal_mask,
            ["movieNm", "예측 총 관객"]
        ].to_numpy(),
        hovertemplate=(
            "영화: %{customdata[0]}<br>"
            "실제 총 관객: %{x:,.0f}명<br>"
            "예측 총 관객: %{customdata[1]:,.0f}명"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 1,000명 미만 예측값
# ---------------------------------------------------------
if low_prediction_count > 0:

    fig.add_trace(
        go.Scatter(
            x=actual_plot[low_mask],
            y=prediction_plot[low_mask],
            mode="markers",
            name="1,000명 미만 예측",
            marker=dict(
                symbol="triangle-down",
                size=10
            ),
            customdata=result.loc[
                low_mask,
                ["movieNm", "예측 총 관객"]
            ].to_numpy(),
            hovertemplate=(
                "영화: %{customdata[0]}<br>"
                "실제 총 관객: %{x:,.0f}명<br>"
                "실제 예측값: %{customdata[1]:,.0f}명"
                "<extra></extra>"
            )
        )
    )


# ---------------------------------------------------------
# 실제값 = 예측값 대각선
# ---------------------------------------------------------
positive_values = np.concatenate(
    [
        actual_plot,
        np.maximum(prediction_raw, 1)
    ]
)

line_min = max(
    positive_values.min(),
    1
)

line_max = positive_values.max()

fig.add_trace(
    go.Scatter(
        x=[line_min, line_max],
        y=[line_min, line_max],
        mode="lines",
        name="실제값 = 예측값",
        line=dict(
            dash="dash",
            width=2
        ),
        hoverinfo="skip"
    )
)


# 로그축
fig.update_xaxes(
    type="log",
    title="실제 총 관객 수 (명)"
)

fig.update_yaxes(
    type="log",
    title="예측한 총 관객 수 (명)"
)


fig.update_layout(
    height=650,
    hovermode="closest"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 1,000명 미만 예측 정보
# =========================================================
st.metric(
    "1,000명보다 작게 예측된 영화",
    f"{low_prediction_count}편"
)

if low_prediction_count > 0:

    st.caption(
        "그래프에서는 로그축을 사용할 수 있도록 "
        "1,000명 미만의 예측값을 그래프의 바닥으로 붙여 표시했습니다. "
        "표에서는 실제 계산된 예측값을 확인할 수 있습니다."
    )
