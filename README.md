# RL for 3-Cushion Billiards (PPO)

본 프로젝트는 PPO (Proximal Policy Optimization)알고리즘을 사용하여,  
물리 엔진 기반의 3쿠션 당구 환경에서 에이전트가 득점 경로를 스스로 학습하도록 하는 강화학습 연구 프로젝트이다.

---

## Project Overview

- **Goal**  
  - 연속적인 행동 공간(힘, 타격 각도, 당점/회전 등)에서  
    **3쿠션 득점을 최대화하는 최적의 정책(Policy)** 을 학습하는 것.
  - 에이전트가 사람과 유사한 샷 초이스와 **연속 득점(run length)** 을 만들어낼 수 있는지 분석.

- **Environment**
  - **PyGame & Pymunk** 기반의 커스텀 당구 시뮬레이터
  - 3개의 공(수구 + 적구 2개)와 테이블, 쿠션, 마찰, 반발력 등을 근사한 2D 물리 환경
  - Gymnasium 스타일 API (`reset()`, `step()`, `render()`)

- **Algorithm**
  - **Stable Baselines 3 (SB3)** 의 **PPO 에이전트**
  - 연속 행동 공간을 위한 Gaussian 정책 + Tanh squash 구조

- **Key Challenges**
  - **Sparse Reward (희소 보상)**  
    - 3쿠션 득점이라는 이벤트가 드물게 발생 → 학습 초기 신호 부족
  - **Policy Collapse (정책 붕괴)**  
    - 학습 후반부에 특정 행동에 과도하게 수렴 → 성능 급락 현상 방지/복구가 필요

---

## Tech Stack

- **Language**
  - Python 3.10.19

- **RL / DL**
  - [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
  - PyTorch

- **Physics / Rendering**
  - PyGame
  - Pymunk

- **Logging / Visualization**
  - Matplotlib
  - TensorBoard (선택)

---

## Installation

### 1. Repository Clone

```bash
git clone https://github.com/KimHyoungchan/RL_3cushion_Agent.git
```

### 2. Dependencies 설치

가상환경 사용

```bash
pip install -r requirements.txt
```

`requirements.txt`가 없다면 기본 의존성은 아래와 같다.

```bash
pip install   gymnasium   stable-baselines3   shimmy   pygame   pymunk   matplotlib   tensorboard
```

---

## Usage
## 0. 학습 모델 다운로드
- 링크 : https://drive.google.com/drive/folders/1GViZ4zEyzjMfsshESQsR14QD-9iOHYJH?usp=drive_link
- **ppo_billiards_3cushion.zip** 혹은 **ppo_billiards_3cushion(SOTA_0.55).zip** 다운로드
- play.py와 같은 디렉토리에 저장
  
### 1. Training (From Scratch)

처음부터 학습을 시작할 때:

- `play.py` 내에서 `TRAIN_MODE = True` 로 설정

```bash
python play.py
```

예시 (내부 로직 기준):

```python
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from environment_set import BilliardEnv
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: BilliardEnv(render_mode=None)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # 하이퍼파라미터들 (예시)
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    model.learn(total_timesteps=500_000)
    model.save("logs/ppo_billiards_3cushion")
```

---

### 2. Resume Training (Manual Fine-tuning)

**핵심 기능**:  
학습 도중 또는 장시간 학습 후 성능이 떨어졌을 때,  
특정 체크포인트를 로드해서 **학습률(LR) 등 하이퍼파라미터를 수정 후 재학습**하는 기능이다.  
이는 특히 **Policy Collapse(정책 붕괴)** 발생 시 정책을 복구하거나 더 안정적으로 fine tuning하기 위한 구현이이다.

```python
from stable_baselines3 import PPO
from environment_set import BilliardEnv
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: BilliardEnv(render_mode=None)])

# 기존 학습된 모델 로드
model = PPO.load("logs/ppo_billiards_3cushion.zip", env=env)

# 하이퍼파라미터 강제 주입 (Fine-tuning)
NEW_LR = 3e-5  # 기존 3e-4 → 1/10로 감소
for param_group in model.policy.optimizer.param_groups:
    param_group["lr"] = NEW_LR

# 추가 학습
model.learn(total_timesteps=100_000)
model.save("logs/ppo_billiards_3cushion")
```

---

### 3. Watching / Evaluation (Policy 테스트)

학습된 정책이 실제로 어떻게 치는지 시각적으로 확인:

```bash
python simulation.py
```

예시 코드 구조:

```python
from stable_baselines3 import PPO
from environment_set import BilliardEnv
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: BilliardEnv(render_mode="human")])
model = PPO.load("logs/ppo_billiards_3cushion", env=env)

obs, _ = env.reset()
for step in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, _ = env.reset()
```

---

## Key Features & Techniques

### 1. Reward Normalization (Custom Scaling)

> 현재 구현은 단순 Min-Max 스케일링 대신, **Stable Baselines 3의 VecNormalize로 Reward를 Normalize**하는 방식이다.  

- **목적**
  - reward scaling(예: 실패 -7, 성공 +55)의 차이가 너무 크면  
    Critic loss가 쉽게 폭발하거나 학습이 불안정해질 수 있음.
  - 학습 안정성 확보: 보상 분포를 평균 0, 분산 1에 가깝게 유지하여 최적화 과정을 평탄하게 만듭니다.
- **방법**
  - 환경 내부에서 `_calculate_reward()`로 **원시 보상(raw_reward)** 를 계산  
    (3쿠션 성공, 파울, 거리 개선 등)
  - `play.py`에서:
    ```python
    # 1. Raw Reward 로깅을 위해 Monitor를 가장 안쪽에 배치
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])

    # 2. 이동 평균/분산을 추적하여 자동 정규화 (보상 클리핑 +/- 10.0 포함)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.)
    ```
  - PPO는 정규화된 보상(normalized_reward)으로 학습
  - **TensorBoard**에는 Monitor가 기록한 실제 점수(reward)를 남겨 직관적인 분석

- **장점**
  - 동적 적응 (Dynamic Adaptation): 학습 초기(낮은 점수)와 후기(높은 점수)의 보상 분포가 달라져도, 
  이동 통계(Running Statistics)가 자동으로 업데이트되어 신경망에 일정한 범위의 신호를 제공.
  - 코드 분리: 환경 코드(step)를 수정하지 않고, 학습 파이프라인에서 정규화를 처리하여 유지보수가 용이함.

---

### 2. Stabilization Strategy

PPO 학습 안정성을 위한 전략들:

- **KL Divergence Monitoring**
  - `target_kl` 값을 기준으로, 한 업데이트에서 KL이 급격히 증가하면  
    → 해당 학습 루프를 조기 종료(Early Stopping)  
    → 정책이 갑자기 너무 많이 바뀌는 것을 방지

- **LR Scheduling**
  - 초기에는 비교적 큰 learning rate(0.0003)로 exploration을 유도
  - 이후 학습이 진행될수록 `lr → decay`시키면서  
    **fine-tuning 구간에서의 진동/발산**을 줄임
  - ex: `linear_schedule(initial_lr)` 형태의 스케줄 사용

---

## Research / Analysis 방향 (예시)

- **학습 안정성 및 붕괴 분석 (Stability & Collapse Analysis)**
  - 희소 보상(Jackpot +55) 발생 시점 전후의 KL Divergence 변화 추적.
  - High LR($3 \times 10^{-4}$) vs Low LR($3 \times 10^{-5}$) 설정에서의 Policy Collapse(성능 급락) 발생 빈도 비교.

- **Normalization 효과 검증**
  - VecNormalize 적용 유무에 따른 Critic Loss의 Variance 및 수렴 속도 비교.
  - Raw Reward와 Normalized Reward의 분포 차이 시각화.

- **상대 좌표 도입 효과 검증**
  - Case A (Absolute): 공의 절대 좌표(x, y)만 제공 (6차원).
  - Case B (Relative): 수구와 목적구 사이의 상대 벡터(거리, 방향) 추가 (11차원).
  - 분석: Case B가 Agent의 Aiming 학습 속도와 수렴 성능(Converge Rate)을 유의미하게 앞당김을 확인.

---

## Troubleshooting Guide

### 1. PyGame 창이 안 뜨거나 바로 꺼짐

- 증상:
  - `render_mode="human"` 으로 실행했는데 창이 바로 꺼짐
- 확인 사항:
  - WSL(Windows Subsystem for Linux) 환경에서는 GUI가 바로 안 뜰 수 있음 → Native Windows Python 권장
  - `render_mode=None` 으로 학습, `render_mode="human"`은 관전 모드에서만 사용
- 권장:
  - 가급적이면 python simulation.py로 시뮬레이션 확인

### 2. GPU를 사용하는 경우

- PyTorch에서 GPU 인식 여부 확인:

  ```python
  import torch
  print(torch.cuda.is_available())
  ```

- `False`일 경우:
  - CUDA 드라이버/Toolkit 설치 여부 확인
  - Colab 사용 시 `런타임 → 런타임 유형 변경 → GPU` 설정 확인

### 3. 학습이 전혀 안 되고 Reward가 계속 0 또는 음수만 나오는 경우

- 점검 포인트:
  - 보상 설계 확인:  
    - 3쿠션 성공 시 보상이 실제로 양수로 들어오는지 (`reward > 0`)  
    - 실패 시에도 **너무 큰 음수**로 패널티를 주지 않았는지 확인
  - action space 확인:
    - `Box(low=-1, high=1, shape=(n,))` 형태에서  
      실제 물리 변수(힘, 각도)로 변환하는 코드가 올바른지

### 4. Policy Collapse (성능이 갑자기 급락)

- 증상:
  - 일정 단계까지 평균 reward가 오르다가  
    어느 순간부터 특정 행동에 고정되고, 성능이 급격히 떨어짐.
- 대응:
  - **학습률 낮추기**: 기존 `3e-4` → `3e-5` 또는 더 작게
  - **checkpoint 롤백**:
    - `logs/ppo_billiards_3cushion.zip` 또는 이전 checkpoint를 로드
    - 위에서 설명한 fine-tuning 방식으로 다시 학습
  - **target_kl 설정**:
    - 안정적이고 보수적인 학습 설정

---

## Repository Structure

```text
RL_3cushion_Agent/
├── _pycache_/               
├── logs/  
├── environment_set.py     
├── simulation.py          
├── play.py                               
├── README.md
└── requirements.txt
```

---
