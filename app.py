import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import io
import os
from datetime import datetime

st.set_page_config(layout="wide")

st.title("❄️ 환경 데이터 분석 및 의사 결정 도우미")
st.markdown("---")

# 0. OpenAI API 키 입력 받기 (사이드바에 배치)
st.sidebar.header("API 키 설정")
openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요", type="password")

client = None
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        # API 키 유효성 간단 테스트
        test_response = client.models.list()
        st.sidebar.success("OpenAI API 키가 성공적으로 설정되었습니다!")
    except Exception as e:
        st.sidebar.error(f"API 키 설정 중 오류가 발생했습니다: {e}")
        client = None
else:
    st.sidebar.warning("OpenAI API 키를 입력해주세요.")
    
st.sidebar.markdown("---")

# 1. 데이터셋 업로드
st.sidebar.header("데이터셋 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드해주세요.", type=["csv"])

# 샘플 데이터셋 경로 개선
sample_data_options = {
    "샘플: 북극 해빙 면적 데이터": "data/N_seaice_extent_daily_v3.0.csv",
}

st.sidebar.markdown("---")
st.sidebar.info("파일이 없으시면 아래 샘플 데이터셋을 선택하여 테스트할 수 있습니다.")
selected_sample = st.sidebar.selectbox("또는 샘플 데이터셋 선택", [""] + list(sample_data_options.keys()))

def load_data():
    """데이터 로딩 함수 (에러 처리 강화)"""
    df = None
    
    if uploaded_file is not None:
        try:
            # 파일 인코딩 문제 대응
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.sidebar.success(f"'{uploaded_file.name}' 파일 업로드 성공!")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='cp949')
                st.sidebar.success(f"'{uploaded_file.name}' 파일 업로드 성공! (CP949 인코딩)")
            except Exception as e:
                st.sidebar.error(f"파일 인코딩 오류: {e}")
        except Exception as e:
            st.sidebar.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            
    elif selected_sample:
        sample_path = sample_data_options[selected_sample]
        if os.path.exists(sample_path):
            try:
                df = pd.read_csv(sample_path)
                st.sidebar.success(f"'{selected_sample}' 데이터셋 로드 성공!")
            except Exception as e:
                st.sidebar.error(f"샘플 데이터셋을 읽는 중 오류: {e}")
        else:
            st.sidebar.warning(f"샘플 파일을 찾을 수 없습니다: {sample_path}")
            st.sidebar.info("직접 CSV 파일을 업로드해주세요.")
    
    return df

def process_date_columns(df):
    """날짜 컬럼 처리 함수 (개선된 로직)"""
    if df is None:
        return df, False
    
    date_col_found = False
    
    # 1. 기존 날짜 컬럼 찾기
    date_candidates = ['Date', 'date', 'DATE', 'Time', 'time', 'TIME', 'datetime', 'Datetime']
    for col in date_candidates:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df = df.dropna(subset=[col])  # 변환 실패한 행 제거
                if not df.empty:
                    df = df.sort_values(col)
                    st.info(f"'{col}' 컬럼을 날짜 형식으로 인식했습니다.")
                    df['Date'] = df[col]
                    date_col_found = True
                    break
            except Exception:
                continue
    
    # 2. 년/월/일 분리된 컬럼으로 날짜 생성
    if not date_col_found:
        ymd_combinations = [
            ('Year', 'Month', 'Day'), ('year', 'month', 'day'), 
            ('YEAR', 'MONTH', 'DAY'), ('Y', 'M', 'D')
        ]
        
        for y, m, d in ymd_combinations:
            if y in df.columns and m in df.columns:
                try:
                    if d in df.columns:
                        df['Date'] = pd.to_datetime(df[[y, m, d]], errors='coerce')
                    else:
                        # 일(Day) 컬럼이 없으면 1일로 설정
                        df['Date'] = pd.to_datetime(df[y].astype(str) + '-' + df[m].astype(str) + '-01', errors='coerce')
                    
                    df = df.dropna(subset=['Date'])
                    if not df.empty:
                        df = df.sort_values('Date')
                        cols_used = f"'{y}', '{m}'" + (f", '{d}'" if d in df.columns else "")
                        st.info(f"{cols_used} 컬럼을 사용하여 'Date' 컬럼을 생성했습니다.")
                        date_col_found = True
                        break
                except Exception:
                    continue
    
    return df, date_col_found

def create_visualizations(df, date_col_found):
    """시각화 생성 함수 (에러 처리 강화)"""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.info("시각화할 수 있는 수치 데이터가 없습니다.")
            return
        
        if date_col_found and 'Date' in df.columns and len(numeric_cols) > 0:
            st.write("시간 경과에 따른 주요 수치 데이터 변화 추이:")
            
            # 적절한 플롯 컬럼 선택
            priority_cols = ['Extent', 'Area', 'CO2', 'Anomaly', 'Temperature', 'Value']
            plot_col = None
            
            for col in priority_cols:
                if col in numeric_cols:
                    plot_col = col
                    break
            
            if plot_col is None and numeric_cols:
                plot_col = numeric_cols[0]

            if plot_col and len(df) > 1:
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # 기본 라인 플롯
                    sns.lineplot(x='Date', y=plot_col, data=df, ax=ax, label=f'{plot_col} 값')
                    
                    # 추세선 추가 (데이터가 충분할 때만)
                    if len(df) > 10:
                        try:
                            x_numeric = df['Date'].apply(lambda date: date.toordinal())
                            sns.regplot(x=x_numeric, y=df[plot_col], ax=ax, scatter=False, 
                                      color='red', line_kws={'linestyle': '--'}, label='추세선')
                        except Exception:
                            pass  # 추세선 그리기 실패해도 기본 그래프는 유지

                    ax.set_title(f'시간 경과에 따른 {plot_col} 변화')
                    ax.set_xlabel('날짜')
                    ax.set_ylabel(plot_col)
                    plt.xticks(rotation=45)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.caption(f"이 그래프는 {plot_col}이 시간 경과에 따라 어떻게 변화했는지 보여줍니다.")

                    # 월별 분석 (Date 컬럼에서 월 추출)
                    try:
                        df['Month_Name'] = df['Date'].dt.strftime('%b')
                        monthly_avg = df.groupby('Month_Name')[plot_col].mean()
                        
                        # 월 순서 정렬
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        monthly_avg = monthly_avg.reindex([m for m in month_order if m in monthly_avg.index])
                        
                        if len(monthly_avg) > 1:
                            st.write(f"월별 평균 {plot_col} (계절성 패턴):")
                            fig2, ax2 = plt.subplots(figsize=(10, 5))
                            sns.barplot(x=monthly_avg.index, y=monthly_avg.values, ax=ax2, palette='viridis')
                            ax2.set_title(f'월별 평균 {plot_col} (계절성)')
                            ax2.set_xlabel('월')
                            ax2.set_ylabel(f'평균 {plot_col}')
                            plt.tight_layout()
                            st.pyplot(fig2)
                            st.caption(f"이 그래프는 연간 {plot_col}의 계절적 변동을 보여줍니다.")
                    except Exception as e:
                        st.warning(f"월별 분석 중 오류가 발생했습니다: {e}")
                        
                except Exception as e:
                    st.error(f"시각화 생성 중 오류가 발생했습니다: {e}")
            else:
                st.info("시각화를 위한 충분한 데이터가 없습니다.")
        else:
            st.info("날짜/시간 컬럼과 수치 컬럼이 모두 존재해야 시계열 시각화를 생성할 수 있습니다.")
    
    except Exception as e:
        st.error(f"시각화 처리 중 오류가 발생했습니다: {e}")

# 메인 로직
df = load_data()

if df is not None:
    df, date_col_found = process_date_columns(df)

st.markdown("---")

# API 키와 데이터 프레임이 모두 준비되었을 때만 주 기능 활성화
if client and df is not None:
    st.subheader("📊 업로드된 데이터 미리보기")
    st.write(df.head())
    st.write(f"데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")
    
    if date_col_found:
        date_range = f"데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
        st.info(date_range)

    st.subheader("❓ 데이터에 대한 질문 입력")
    user_question = st.text_area("업로드된 데이터에 대해 궁금한 점을 질문해주세요:",
                                 placeholder="예: '이 데이터셋에서 해빙 면적의 연간 평균 변화 추세는 어떻게 되나요?', '가장 큰 변화를 보인 기간은 언제인가요?', '이러한 환경 변화가 생태계에 미칠 잠재적 영향은 무엇인가요?'")

    if st.button("답변 생성"):
        if user_question:
            with st.spinner("GPT가 데이터를 분석 중입니다..."):
                try:
                    # 데이터 요약 정보 준비
                    data_head = df.head().to_markdown(index=False)
                    data_description = df.describe().to_markdown()
                    
                    buffer = io.StringIO() 
                    df.info(buf=buffer, verbose=True, show_counts=True)
                    column_info_str = buffer.getvalue()

                    time_range = ""
                    if date_col_found and 'Date' in df.columns:
                        time_range = f"데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
                    
                    prompt = f"""
                    당신은 기후 변화 및 환경 데이터 분석 전문가입니다. 주어진 환경 데이터에 대한 사용자의 질문에 답하고,
                    필요하다면 시각화를 위한 제안과 환경 변화에 대응하기 위한 의사 결정 또는 정책적 인사이트를 제공해주세요.

                    데이터 요약 (첫 5행):
                    {data_head}

                    데이터 통계 요약:
                    {data_description}

                    컬럼 정보:
                    {column_info_str}

                    {time_range}

                    사용자의 질문: "{user_question}"

                    답변은 다음 형식으로 구성해주세요:
                    1. **환경 데이터 분석 결과:** 질문에 대한 직접적인 데이터 기반 답변
                    2. **시각화 제안 (선택 사항):** 답변을 뒷받침하기 위한 시각화 아이디어
                    3. **의사 결정 및 정책 인사이트:** 분석 결과를 바탕으로 한 구체적인 제안
                    """

                    # OpenAI API 호출 (모델명 수정)
                    response = client.chat.completions.create(
                        model="gpt-4",  # 또는 "gpt-3.5-turbo"
                        messages=[
                            {"role": "system", "content": "You are a helpful climate and environmental data analysis expert."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    gpt_response = response.choices[0].message.content
                    st.subheader("✨ GPT의 분석 결과 및 의사 결정 지원")
                    st.markdown(gpt_response)

                    st.markdown("---")
                    st.subheader("📈 주요 시각화")
                    
                    create_visualizations(df, date_col_found)

                except Exception as e:
                    st.error(f"GPT API 호출 중 오류가 발생했습니다: {e}")
                    st.info("API 키가 유효한지, 사용 한도가 남아있는지 확인해주세요.")
        else:
            st.warning("질문을 입력해주세요!")

else:
    if not openai_api_key:
        st.info("🔑 왼쪽 사이드바에 OpenAI API 키를 입력해주세요.")
    elif df is None:
        st.info("📁 왼쪽 사이드바에서 CSV 파일을 업로드하거나 샘플 데이터셋을 선택해주세요.")

st.markdown("---")
st.sidebar.markdown("💡 이 앱은 업로드된 환경 데이터 분석을 통해 기후 변화에 대한 의사 결정을 돕기 위해 GPT를 활용합니다.")
st.sidebar.markdown("🏫 중고등학교 환경 교육에도 활용 가능합니다.")
