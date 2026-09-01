"""원본 LP 데이터(xlsx/csv)를 pandas DataFrame으로 읽어오는 모듈.

TODO:
- read_lp_excel(path) -> pd.DataFrame  : 헤더가 병합셀인 경우(예: '수전' 그룹 헤더) 처리
- read_lp_csv(path) -> pd.DataFrame
- 여러 파일(월별 등)을 합치는 concat_lp_files(paths)
"""
