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
        
        # 데이터 구조 정보
        data_info = {
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "data_size": f"{df.shape[0]} 행, {df.shape[1]} 열",
            "has_date": date_col_found,
            "sample_data": df.head(3).to_dict()
        }
        
        if date_col_found and 'Date' in df.columns:
            data_info["date_range"] = f"{df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
        
        prompt = f"""
        사용자가 환경 데이터에 대해 다음과 같은 질문을 했습니다: "{user_question}"

        데이터 구조:
        {json.dumps(data_info, ensure_ascii=False, indent=2)}

        이 데이터와 질문에 가장 적합한 시각화 2개를 추천해주세요. 
        각 시각화에 대해 다음 형식의 JSON으로 응답해주세요:

        {{
            "visualization_1": {{
                "type": "line_plot/bar_plot/scatter_plot/histogram",
                "title": "그래프 제목",
                "x_column": "x축 컬럼명",
                "y_column": "y축 컬럼명",
                "description": "이 시각화가 보여주는 내용과 의미"
            }},
            "visualization_2": {{
                "type": "line_plot/bar_plot/scatter_plot/histogram",
                "title": "그래프 제목", 
                "x_column": "x축 컬럼명",
                "y_column": "y축 컬럼명",
                "description": "이 시각화가 보여주는 내용과 의미"
            }}
        }}

        주의사항:
        - x_column과 y_column은 실제 데이터에 존재하는 컬럼명을 사용하세요
        - 환경 데이터 분석에 의미 있는 시각화를 추천하세요
        - JSON 형식만 응답하고 다른 텍스트는 포함하지 마세요
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in environmental data visualization. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        viz_recommendations = json.loads(response.choices[0].message.content)
        return viz_recommendations
        
    except Exception as e:
        st.error(f"시각화 추천 중 오류: {e}")
        return None

def create_visualization_from_recommendation(df, viz_config, viz_num):
    """GPT 추천에 따라 시각화 생성"""
    try:
        viz_type = viz_config.get('type', 'line_plot')
        title = viz_config.get('title', f'시각화 {viz_num}')
        x_col = viz_config.get('x_column')
        y_col = viz_config.get('y_column')
        description = viz_config.get('description', '')
        
        # 컬럼 존재 확인
        if x_col not in df.columns or y_col not in df.columns:
            st.warning(f"시각화 {viz_num}: 추천된 컬럼({x_col}, {y_col})이 데이터에 없습니다.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == 'line_plot':
            if x_col == 'Date' or 'date' in x_col.lower():
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
            else:
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
                
        elif viz_type == 'bar_plot':
            if len(df[x_col].unique()) > 20:  # 너무 많은 카테고리가 있으면 상위 20개만
                top_values = df.nlargest(20, y_col)
                sns.barplot(data=top_values, x=x_col, y=y_col, ax=ax)
                plt.xticks(rotation=45)
            else:
                sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
                plt.xticks(rotation=45)
                
        elif viz_type == 'scatter_plot':
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, alpha=0.6)
            
        elif viz_type == 'histogram':
            sns.histplot(data=df, x=y_col, ax=ax, bins=30)
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        st.caption(f"**{title}**: {description}")
        
    except Exception as e:
        st.error(f"시각화 {viz_num} 생성 중 오류: {e}")

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
                viz_recommendations = get_visualization_recommendations(df, user_question, date_col_found)
                
                if viz_recommendations:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 시각화 1")
                        if 'visualization_1' in viz_recommendations:
                            create_visualization_from_recommendation(df, viz_recommendations['visualization_1'], 1)
                    
                    with col2:
                        st.markdown("### 📊 시각화 2")
                        if 'visualization_2' in viz_recommendations:
                            create_visualization_from_recommendation(df, viz_recommendations['visualization_2'], 2)
                
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
