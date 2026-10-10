
import re
from itertools import combinations

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/teacher188-netizen/"
    "2026-datascience/refs/heads/main/sangok_meals.csv"
)

st.set_page_config(
    page_title="급식 규칙 찾기",
    page_icon="🍱",
    layout="wide",
)

st.title("🍱 급식 규칙 찾기")
st.caption("우리 학교 급식에서 함께 나오는 메뉴의 규칙을 찾아봅니다.")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = {"날짜", "식사", "메뉴"}
    if not required.issubset(df.columns):
        raise ValueError("CSV에 날짜, 식사, 메뉴 열이 필요합니다.")

    return df[["날짜", "식사", "메뉴"]].copy()


def clean_menu(name):
    """메뉴 이름에서 괄호와 괄호 안의 내용 제거"""
    name = str(name).strip()

    # 일반 괄호와 전각 괄호 모두 처리
    previous = None
    while previous != name:
        previous = name
        name = re.sub(r"\([^()]*\)", "", name)
        name = re.sub(r"\（[^（）]*\）", "", name)

    return re.sub(r"\s+", " ", name).strip()


def make_basket(menu_text):
    """하루 식단을 메뉴 집합으로 변환"""
    return {
        item
        for raw in str(menu_text).split("|")
        if (item := clean_menu(raw))
    }


try:
    data = load_data()
except Exception as e:
    st.error(f"급식 데이터를 불러오지 못했습니다: {e}")
    st.stop()


# 중식 데이터만 선택
lunch = data[
    data["식사"].fillna("").astype(str).str.strip().str.contains(
        "중식", na=False
    )
].copy()

# 중식 표시 행이 없으면 CSV 전체 행 사용
if lunch.empty:
    lunch = data.copy()
    st.info("'중식' 행이 없어 CSV 전체 데이터를 분석합니다.")

lunch["장바구니"] = lunch["메뉴"].fillna("").apply(make_basket)
lunch = lunch[lunch["장바구니"].map(bool)].reset_index(drop=True)

n_days = len(lunch)

if n_days == 0:
    st.warning("분석할 급식 데이터가 없습니다.")
    st.stop()


# 메뉴별 등장 일수와 메뉴 쌍별 동시 등장 일수
menu_counts = {}
pair_counts = {}

for basket in lunch["장바구니"]:
    for menu in basket:
        menu_counts[menu] = menu_counts.get(menu, 0) + 1

    for a, b in combinations(sorted(basket), 2):
        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1


# 모든 메뉴 쌍의 양방향 연관 규칙 생성
rules = []

for (a, b), both in pair_counts.items():
    support = both / n_days
    support_a = menu_counts[a] / n_days
    support_b = menu_counts[b] / n_days

    # A → B
    rules.append({
        "조건": a,
        "결과": b,
        "동시": both,
        "지지도": support,
        "신뢰도": both / menu_counts[a],
        "향상도": (both / menu_counts[a]) / support_b,
    })

    # B → A
    rules.append({
        "조건": b,
        "결과": a,
        "동시": both,
        "지지도": support,
        "신뢰도": both / menu_counts[b],
        "향상도": (both / menu_counts[b]) / support_a,
    })

rules_df = pd.DataFrame(
    rules,
    columns=["조건", "결과", "동시", "지지도", "신뢰도", "향상도"],
)


# 상단 요약
c1, c2, c3 = st.columns(3)
c1.metric("급식 일수", f"{n_days:,}일")
c2.metric("메뉴 종류 수", f"{len(menu_counts):,}개")
c3.metric("함께 나온 메뉴 쌍 수", f"{len(pair_counts):,}쌍")

st.divider()


# 최소 동시 일수 슬라이더
st.subheader("🔎 연관 규칙 필터")

min_cooccurrence = st.slider(
    "최소 동시 일수",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="선택한 일수보다 적게 함께 나온 메뉴 규칙은 표와 그래프에서 제외합니다.",
)


# 메뉴 필터 및 정렬
menus = sorted(menu_counts.keys())

col1, col2 = st.columns([2, 1])

with col1:
    selected_menu = st.selectbox(
        "좁혀 보기 메뉴",
        ["전체 메뉴"] + menus,
        help="선택한 메뉴가 조건 또는 결과에 포함된 규칙만 표시합니다.",
    )

with col2:
    sort_option = st.selectbox(
        "정렬 기준",
        ["향상도 순", "신뢰도 순", "동시 순"],
    )


# 최소 동시 일수 적용
filtered = rules_df[
    rules_df["동시"] >= min_cooccurrence
].copy()

# 선택 메뉴 적용
if selected_menu != "전체 메뉴":
    filtered = filtered[
        (filtered["조건"] == selected_menu)
        | (filtered["결과"] == selected_menu)
    ]

# 정렬
sort_column = {
    "향상도 순": "향상도",
    "신뢰도 순": "신뢰도",
    "동시 순": "동시",
}[sort_option]

filtered = filtered.sort_values(
    [sort_column, "신뢰도", "동시"],
    ascending=[False, False, False],
).reset_index(drop=True)


# 필터링 후 남은 규칙 수 표시
st.write(f"**현재 조건에 맞는 규칙: {len(filtered):,}개**")


# 연관 규칙 표
st.subheader("📋 연관 규칙 표")
st.caption(
    "지지도와 신뢰도는 백분율(%), 향상도는 배수입니다. "
    "동시는 두 메뉴가 함께 나온 급식 일수입니다."
)

if filtered.empty:
    st.info("선택한 조건에 맞는 규칙이 없습니다.")
else:
    table = filtered[
        ["조건", "결과", "동시", "지지도", "신뢰도", "향상도"]
    ].copy()

    table["지지도"] = table["지지도"].map(lambda x: f"{x:.2%}")
    table["신뢰도"] = table["신뢰도"].map(lambda x: f"{x:.2%}")
    table["향상도"] = table["향상도"].map(lambda x: f"{x:.3f}")

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# 향상도 상위 10개 그래프
st.subheader("📊 향상도 상위 10개 규칙")

if filtered.empty:
    st.info("그래프로 표시할 규칙이 없습니다.")
else:
    top10 = filtered.sort_values(
        ["향상도", "신뢰도", "동시"],
        ascending=[False, False, False],
    ).head(10).copy()

    top10["규칙"] = top10["조건"] + " → " + top10["결과"]

    chart_data = top10.sort_values("향상도", ascending=True).copy()
    chart_data["향상도 표시"] = chart_data["향상도"].map(
        lambda x: f"{x:.3f}"
    )

    chart = px.bar(
        chart_data,
        x="향상도",
        y="규칙",
        orientation="h",
        text="향상도 표시",
        labels={
            "향상도": "향상도 (배)",
            "규칙": "연관 규칙",
        },
        hover_data={
            "조건": True,
            "결과": True,
            "동시": True,
            "지지도": ":.2%",
            "신뢰도": ":.2%",
            "향상도": ":.3f",
            "향상도 표시": False,
        },
    )

    chart.update_traces(textposition="outside", cliponaxis=False)
    chart.update_layout(
        xaxis_title="향상도 (배)",
        yaxis_title="",
        margin=dict(l=20, r=70, t=30, b=20),
    )

    st.plotly_chart(chart, use_container_width=True)


# 함께 나온 적 없는 짝
st.divider()
st.subheader("🚫 함께 나온 적 없는 짝")
st.write(
    "선택한 메뉴와 같은 날 한 번도 함께 나오지 않았지만, "
    "각각 10일 이상 등장한 메뉴를 보여 줍니다."
)

selected_menu_absent = st.selectbox(
    "함께 나온 적 없는 짝을 찾을 메뉴",
    menus,
    key="absent_pair_menu",
    help="위의 '좁혀 보기 메뉴'와 별도로 선택할 수 있습니다.",
)

selected_days = menu_counts[selected_menu_absent]
st.write(
    f"**선택한 메뉴 '{selected_menu_absent}'의 등장 일수: "
    f"{selected_days:,}일**"
)

# 선택 메뉴와 함께 나온 적이 있는 메뉴 집합
cooccurred_menus = set()

for (a, b), count in pair_counts.items():
    if count > 0:
        if a == selected_menu_absent:
            cooccurred_menus.add(b)
        elif b == selected_menu_absent:
            cooccurred_menus.add(a)

# 10일 이상 등장했지만 선택 메뉴와 동시 등장한 적 없는 메뉴
absent_rows = []

for menu, count in menu_counts.items():
    if menu == selected_menu_absent:
        continue

    if count < 10:
        continue

    if menu in cooccurred_menus:
        continue

    absent_rows.append({
        "메뉴": menu,
        "등장 일수": count,
    })

absent_df = pd.DataFrame(
    absent_rows,
    columns=["메뉴", "등장 일수"],
)

if absent_df.empty:
    st.info(
        "이 메뉴와 한 번도 함께 나오지 않았고, "
        "10일 이상 등장한 다른 메뉴가 없습니다."
    )
else:
    absent_df = absent_df.sort_values(
        ["등장 일수", "메뉴"],
        ascending=[False, True],
    ).reset_index(drop=True)

    st.write(f"**조건에 맞는 메뉴: {len(absent_df):,}개**")

    st.dataframe(
        absent_df,
        use_container_width=True,
        hide_index=True,
    )


st.caption(
    "분석 기준: 하루 식단을 장바구니 하나로 계산합니다. "
    "괄호와 괄호 안의 내용을 제거하며, 같은 식단 안의 중복 메뉴는 "
    "한 번만 계산합니다."
)
