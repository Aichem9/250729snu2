import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import io
import os
from datetime import datetime
import json

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
                df = df.dropna(subset=[col])
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

def get_visualization_recommendations(df, user_question, date_col_found):
    """GPT로부터 시각화 추천을 받는 함수"""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 데이터 구조 정보 간소화
        data_info = f"""
        컬럼: {list(df.columns)}
        수치형 컬럼: {numeric_cols}
        범주형 컬럼: {categorical_cols}
        데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열
        날짜 컬럼 존재: {date_col_found}
        """
        
        if date_col_found and 'Date' in df.columns:
            data_info += f"\n날짜 범위: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
        
        prompt = f"""
        사용자 질문: "{user_question}"

        데이터 정보:
        {data_info}

        이 데이터에 적합한 시각화 2개를 JSON으로만 응답하세요:

        {{
            "visualization_1": {{
                "type": "line_plot",
                "title": "시간에 따른 변화",
                "x_column": "{numeric_cols[0] if numeric_cols else 'Date'}",
                "y_column": "{numeric_cols[0] if numeric_cols else list(df.columns)[0]}",
                "description": "데이터의 시간적 변화를 보여줍니다"
            }},
            "visualization_2": {{
                "type": "bar_plot",
                "title": "분포 현황", 
                "x_column": "{categorical_cols[0] if categorical_cols else list(df.columns)[0]}",
                "y_column": "{numeric_cols[0] if numeric_cols else list(df.columns)[1]}",
                "description": "데이터의 분포를 보여줍니다"
            }}
        }}

        위 형식으로 실제 컬럼명을 사용하여 JSON만 응답하세요.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in data visualization. Respond only with valid JSON using actual column names from the data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            # 코드 블록 제거
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            viz_recommendations = json.loads(response_text)
            return viz_recommendations
            
        except json.JSONDecodeError as e:
            st.warning(f"GPT JSON 파싱 실패, 기본 시각화를 생성합니다: {e}")
            # 기본 시각화 반환
            return create_default_visualizations(df, numeric_cols, date_col_found)
        
    except Exception as e:
        st.warning(f"시각화 추천 중 오류, 기본 시각화를 생성합니다: {e}")
        return create_default_visualizations(df, df.select_dtypes(include=['number']).columns.tolist(), date_col_found)

def create_default_visualizations(df, numeric_cols, date_col_found):
    """기본 시각화 설정 반환"""
    if not numeric_cols:
        return None
    
    viz1 = {
        "type": "line_plot",
        "title": f"{numeric_cols[0]} 변화 추이",
        "x_column": "Date" if date_col_found and 'Date' in df.columns else df.columns[0],
        "y_column": numeric_cols[0],
        "description": f"{numeric_cols[0]}의 변화 패턴을 보여줍니다"
    }
    
    viz2 = {
        "type": "histogram",
        "title": f"{numeric_cols[0]} 분포",
        "x_column": numeric_cols[0],
        "y_column": numeric_cols[0],
        "description": f"{numeric_cols[0]}의 분포 현황을 보여줍니다"
    }
    
    if len(numeric_cols) > 1:
        viz2 = {
            "type": "scatter_plot",
            "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
            "x_column": numeric_cols[0],
            "y_column": numeric_cols[1],
            "description": f"{numeric_cols[0]}와 {numeric_cols[1]}의 상관관계를 보여줍니다"
        }
    
    return {
        "visualization_1": viz1,
        "visualization_2": viz2
    }

def create_visualization_from_recommendation(df, viz_config, viz_num):
    """GPT 추천에 따라 시각화 생성"""
    try:
        viz_type = viz_config.get('type', 'line_plot')
        title = viz_config.get('title', f'시각화 {viz_num}')
        x_col = viz_config.get('x_column')
        y_col = viz_config.get('y_column')
        description = viz_config.get('description', '')
        
        # 컬럼 존재 확인 및 대안 제시
        if x_col not in df.columns:
            st.warning(f"컬럼 '{x_col}'이 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                x_col = numeric_cols[0]
            else:
                x_col = df.columns[0]
                
        if y_col not in df.columns:
            st.warning(f"컬럼 '{y_col}'이 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                y_col = numeric_cols[0]
            else:
                y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # matplotlib 설정
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
        
        # 시각화 생성
        if viz_type == 'line_plot':
            if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                df_sorted = df.sort_values(x_col)
                ax.plot(df_sorted[x_col], df_sorted[y_col], marker='o', linewidth=2, markersize=4)
            else:
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
                
        elif viz_type == 'bar_plot':
            if df[x_col].dtype == 'object' or len(df[x_col].unique()) < 20:
                # 범주형 데이터의 경우
                if df[y_col].dtype in ['int64', 'float64']:
                    group_data = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(15)
                    ax.bar(range(len(group_data)), group_data.values)
                    ax.set_xticks(range(len(group_data)))
                    ax.set_xticklabels(group_data.index, rotation=45)
                else:
                    value_counts = df[x_col].value_counts().head(15)
                    ax.bar(range(len(value_counts)), value_counts.values)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45)
            else:
                # 수치형 데이터 히스토그램
                ax.hist(df[x_col].dropna(), bins=20, alpha=0.7)
                
        elif viz_type == 'scatter_plot':
            if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
            else:
                st.warning(f"산점도를 위해서는 수치형 데이터가 필요합니다. ({x_col}: {df[x_col].dtype}, {y_col}: {df[y_col].dtype})")
                return
                
        elif viz_type == 'histogram':
            if pd.api.types.is_numeric_dtype(df[y_col]):
                ax.hist(df[y_col].dropna(), bins=20, alpha=0.7, edgecolor='black')
            else:
                value_counts = df[y_col].value_counts().head(15)
                ax.bar(range(len(value_counts)), value_counts.values)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45)
        
        # 그래프 꾸미기
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # Streamlit에 표시
        st.pyplot(fig)
        st.caption(f"**{title}**: {description}")
        
        # 메모리 정리
        plt.close(fig)
        
    except Exception as e:
        st.error(f"시각화 {viz_num} 생성 중 오류: {e}")
        st.write(f"디버깅 정보 - 시각화 설정: {viz_config}")
        
        # 간단한 대안 시각화 시도
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(df[numeric_cols[0]].head(50), marker='o')
                ax.set_title(f"{numeric_cols[0]} 기본 차트")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
        except:
            st.error("기본 시각화도 생성할 수 없습니다.")

def safe_dataframe_to_text(df, method='head'):
    """tabulate 의존성 없이 데이터프레임을 텍스트로 변환"""
    try:
        if method == 'head':
            return df.head().to_markdown(index=False)
        elif method == 'describe':
            return df.describe().to_markdown()
    except ImportError:
        if method == 'head':
            return df.head().to_string(index=False)
        elif method == 'describe':
            return df.describe().to_string()
    except Exception as e:
        if method == 'head':
            return str(df.head())
        elif method == 'describe':
            return str(df.describe())

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

    if st.button("분석 시작"):
        if user_question:
            with st.spinner("GPT가 데이터를 분석하고 시각화를 생성 중입니다..."):
                
                # 1단계: 시각화 추천 받기
                st.subheader("📈 GPT 추천 시각화")
                
                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("GPT로부터 시각화 추천을 받는 중...")
                progress_bar.progress(25)
                
                viz_recommendations = get_visualization_recommendations(df, user_question, date_col_found)
                progress_bar.progress(50)
                
                if viz_recommendations:
                    status_text.text("시각화를 생성하는 중...")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 시각화 1")
                        if 'visualization_1' in viz_recommendations:
                            create_visualization_from_recommendation(df, viz_recommendations['visualization_1'], 1)
                    
                    progress_bar.progress(75)
                    
                    with col2:
                        st.markdown("### 📊 시각화 2")
                        if 'visualization_2' in viz_recommendations:
                            create_visualization_from_recommendation(df, viz_recommendations['visualization_2'], 2)
                    
                    progress_bar.progress(100)
                    status_text.text("시각화 생성 완료!")
                    
                    # 진행 바 정리
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                else:
                    st.warning("시각화 추천을 받을 수 없어 기본 시각화를 생성합니다.")
                    # 기본 시각화 생성
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 📊 기본 시각화 1")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.plot(df[numeric_cols[0]].head(100), marker='o', markersize=3)
                            ax.set_title(f"{numeric_cols[0]} 변화")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col2:
                            st.markdown("### 📊 기본 시각화 2") 
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.hist(df[numeric_cols[0]].dropna(), bins=20, alpha=0.7)
                            ax.set_title(f"{numeric_cols[0]} 분포")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)
                
                st.markdown("---")
                
                # 2단계: 분석 결과 및 인사이트 생성
                try:
                    data_head = safe_dataframe_to_text(df, 'head')
                    data_description = safe_dataframe_to_text(df, 'describe')
                    
                    buffer = io.StringIO() 
                    df.info(buf=buffer, verbose=True, show_counts=True)
                    column_info_str = buffer.getvalue()

                    time_range = ""
                    if date_col_found and 'Date' in df.columns:
                        time_range = f"데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
                    
                    prompt = f"""
                    당신은 기후 변화 및 환경 데이터 분석 전문가입니다. 주어진 환경 데이터에 대한 사용자의 질문에 답하고,
                    환경 변화에 대응하기 위한 의사 결정 또는 정책적 인사이트를 제공해주세요.

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
                    2. **주요 패턴 및 트렌드:** 데이터에서 발견되는 중요한 패턴이나 변화 추세
                    3. **의사 결정 및 정책 인사이트:** 분석 결과를 바탕으로 한 구체적인 제안 및 대응 방안
                    4. **향후 연구 방향:** 추가로 필요한 데이터나 연구 방향 제시
                    """

                    response = client.chat.completions.create(
                        model="gpt-4",
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

                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")
                    st.info("API 키가 유효한지, 사용 한도가 남아있는지 확인해주세요.")
                    
                    if st.checkbox("상세 오류 정보 보기"):
                        st.exception(e)
        else:
            st.warning("질문을 입력해주세요!")

else:
    if not openai_api_key:
        st.info("🔑 왼쪽 사이드바에 OpenAI API 키를 입력해주세요.")
    elif df is None:
        st.info("📁 왼쪽 사이드바에서 CSV 파일을 업로드해주세요.")

st.markdown("---")
st.sidebar.markdown("💡 이 앱은 업로드된 환경 데이터 분석을 통해 기후 변화에 대한 의사 결정을 돕기 위해 GPT를 활용합니다.")
st.sidebar.markdown("🏫 중고등학교 환경 교육에도 활용 가능합니다.")
