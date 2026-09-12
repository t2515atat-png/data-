# ============================================================
# 그래프 2. 장르 및 영화별 총 관객 수 (트리맵)
# ============================================================
st.divider()
st.header("2. 장르 및 영화별 총 관객 수")

# 총 관객 수(total_audi)가 0보다 큰 데이터만 필터링 (트리맵 오류 방지 및 결측값 처리)
df_treemap = df[df["total_audi"] > 0].copy()

fig2 = px.treemap(
    df_treemap,
    path=["genre", "movieNm"],  # 장르 안에 영화가 들어가는 계층 구조
    values="total_audi",        # 칸의 크기: 총 관객 수
    title="장르 및 영화별 총 관객 수 분포",
)

# 마우스를 올렸을 때 영화명(또는 장르명)과 총 관객 수 표시
fig2.update_traces(
    hovertemplate="<b>%{label}</b><br>총 관객 수: %{value:,}명<extra></extra>"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("**이 그래프로 알 수 있는 것**")
st.text_input(
    "문구를 입력하세요.",
    placeholder="예: 장르별 총 관객 수의 비중과 각 장르 내에서 관객 수 점유율이 높은 대표 영화를 파악할 수 있다.",
    key="graph2_note",
)
