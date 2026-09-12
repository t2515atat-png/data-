import pandas as pd

# 1. 급식 데이터 불러오기
# Excel 파일인 경우: pd.read_excel('파일경로.xlsx')
# CSV 파일인 경우: pd.read_csv('파일경로.csv', encoding='utf-8')
df = pd.read_csv("급식데이터.csv")

# 2. '요리명' 열에서 '자장밥' 문구가 포함된 데이터만 필터링
# na=False는 결측치(빈 칸)가 있어도 에러가 나지 않게 처리합니다.
jajang_df = df[df["요리명"].str.contains("자장밥", na=False)]

# 3. 결과 확인
print(jajang_df)
