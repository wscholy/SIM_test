import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 페이지 설정 ---
st.set_page_config(
    page_title="기후변화 생태계 시뮬레이션",
    page_icon="🌍",
    layout="wide"
)

# --- 제목 및 설명 ---
st.title("🌍 기후변화와 생태계 생존 시뮬레이션")
st.markdown("""
이 대시보드는 **기온 상승**이 **토끼(피식자)**와 **여우(포식자)**의 생태계 균형에 미치는 영향을 시뮬레이션합니다.
기온이 오르면 피식자의 먹이(풀)가 줄어들고, 포식자의 에너지 소모가 늘어나는 시나리오를 가정합니다.
""")

st.divider()

# --- 사이드바: 파라미터 조절 ---
st.sidebar.header("⚙️ 시뮬레이션 설정")

# 1. 기후 변수
st.sidebar.subheader("1. 기후 환경 설정")
temp_rise = st.sidebar.slider(
    "지구 평균 기온 상승 (°C)", 
    min_value=0.0, 
    max_value=5.0, 
    value=0.0, 
    step=0.1,
    help="기온이 상승할수록 토끼의 번식률은 감소하고 여우의 사망률은 증가합니다."
)

# 2. 초기 개체수
st.sidebar.subheader("2. 초기 개체수 설정")
init_rabbits = st.sidebar.number_input("초기 토끼 수 (마리)", value=40, step=1)
init_foxes = st.sidebar.number_input("초기 여우 수 (마리)", value=9, step=1)

# 3. 시뮬레이션 기간
st.sidebar.subheader("3. 시간 설정")
years = st.sidebar.slider("시뮬레이션 기간 (년)", 10, 100, 50)

# --- 모델링 로직 (Lotka-Volterra 수정 모델) ---
def climate_ecosystem_model(z, t, temp_rise):
    """
    R: 토끼(Prey), F: 여우(Predator)
    dR/dt = alpha*R - beta*R*F  (토끼 변화율)
    dF/dt = delta*R*F - gamma*F (여우 변화율)
    """
    R, F = z
    
    # 기본 계수 (건강한 환경 기준)
    alpha_base = 1.1  # 토끼 번식률
    beta = 0.4        # 토끼가 잡아먹히는 비율
    delta = 0.1       # 여우가 토끼를 먹고 번식하는 효율
    gamma_base = 0.4  # 여우 자연 사망률
    
    # 기후 변화 영향 적용 (가설)
    # 1. 기온 상승 -> 식물 감소 -> 토끼 번식률(alpha) 감소
    # 0.1의 계수는 민감도를 의미함
    alpha_climate = alpha_base * (1 - (temp_rise * 0.15))
    
    # 2. 기온 상승 -> 열 스트레스 -> 여우 사망률(gamma) 증가
    gamma_climate = gamma_base * (1 + (temp_rise * 0.1))
    
    # 생태계 붕괴 방지 (음수 방지)
    if alpha_climate < 0: alpha_climate = 0
    
    dRdt = alpha_climate * R - beta * R * F
    dFdt = delta * R * F - gamma_climate * F
    
    return [dRdt, dFdt]

# --- 시뮬레이션 실행 ---
t = np.linspace(0, years, years * 10) # 시간 축 생성
initial_state = [init_rabbits, init_foxes]

# ODE 풀이
result = odeint(climate_ecosystem_model, initial_state, t, args=(temp_rise,))
rabbits = result[:, 0]
foxes = result[:, 1]

# --- 결과 시각화 ---

# 1. 메인 그래프 (시간에 따른 개체수 변화)
st.subheader(f"📈 시뮬레이션 결과 (기온 상승: +{temp_rise}°C)")

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, rabbits, label='토끼 (Prey)', color='blue', linewidth=2)
    ax.plot(t, foxes, label='여우 (Predator)', color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('시간 (년)')
    ax.set_ylabel('개체 수')
    ax.set_title('시간에 따른 생태계 개체수 변화')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# 2. 데이터 지표 및 분석
with col2:
    st.markdown("#### 핵심 지표")
    
    final_rabbit = int(rabbits[-1])
    final_fox = int(foxes[-1])
    
    # 토끼 상태 표시
    delta_rabbit = final_rabbit - init_rabbits
    st.metric(label="최종 토끼 수", value=f"{final_rabbit} 마리", delta=delta_rabbit)
    
    # 여우 상태 표시
    delta_fox = final_fox - init_foxes
    st.metric(label="최종 여우 수", value=f"{final_fox} 마리", delta=delta_fox)
    
    # 생태계 상태 진단
    st.markdown("#### 🔍 생태계 진단")
    if final_rabbit < 2 or final_fox < 2:
        st.error("💀 생태계 붕괴: 멸종 위기종 발생")
    elif temp_rise >= 3.0:
        st.warning("⚠️ 고위험: 생태계가 매우 불안정함")
    else:
        st.success("✅ 안정: 생태계 균형 유지 중")

# 3. 위상 공간 (Phase Plane) - 심화 분석
with st.expander("생태계 위상 공간 (Phase Plane) 보기"):
    st.markdown("토끼와 여우 개체 수의 상관관계를 보여줍니다. 원형 궤도를 그리면 주기적인 안정을 의미합니다.")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(rabbits, foxes, color='green')
    ax2.set_xlabel('토끼 수')
    ax2.set_ylabel('여우 수')
    ax2.set_title('포식자-피식자 위상 공간')
    ax2.grid(True)
    st.pyplot(fig2)

# --- 하단 설명 ---
st.info("""
**💡 시뮬레이션 원리:**
* **기온 0°C 상승 시:** 토끼와 여우는 자연스러운 주기를 그리며 번성합니다.
* **기온 상승 시:** 토끼의 번식률이 떨어지고 여우의 사망률이 올라가며, 그래프의 진폭이 작아지거나 결국 0으로 수렴(멸종)하게 됩니다.
""")
