# RL for 3-Cushion Billiards (PPO)

본 프로젝트는 강화학습 **PPO (Proximal Policy Optimization)**알고리즘을 사용하여,  
물리 엔진 기반의 3쿠션 당구 환경에서 에이전트가 득점 경로를 스스로 학습하도록 하는 연구 프로젝트입니다.

---

## Project Overview

- **Goal**  
  - 연속적인 행동 공간(힘, 타격 각도, 당점/회전 등)에서  
    **3쿠션 득점을 최대화하는 최적의 정책(Policy)** 을 학습하는 것.
  - 에이전트가 사람과 유사한 샷 선택(초이스)과 **연속 득점(run length)** 을 만들어낼 수 있는지 분석.

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
  - Python 3.12

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
git clone https://github.com/KimHyoungchan/RL_3cushion_agent.git
cd billiard-rl-agent
```

### 2. Dependencies 설치

가상환경 사용

```bash
pip install -r requirements.txt
```

`requirements.txt`가 없다면 기본 의존성은 아래와 같습니다.

```bash
pip install   gymnasium   stable-baselines3   shimmy   pygame   pymunk   matplotlib   tensorboard
```

---

## Usage

### 1. Training (From Scratch)

처음부터 학습을 시작할 때:

- `play.py` 내에서 `TRAIN_MODE = True` 로 설정하거나,
- CLI 인자를 사용하는 경우:

```bash
python main.py --train
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
특정 체크포인트를 로드해서 **학습률(LR) 등 하이퍼파라미터를 수정 후 재학습**하는 기능입니다.  
이는 특히 **Policy Collapse(정책 붕괴)** 발생 시 정책을 복구하거나 더 안정적으로 미세 조정할 때 유용합니다.

```python
from stable_baselines3 import PPO
from environment_set import BilliardEnv
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: BilliardEnv(render_mode=None)])

# 기존 학습된 모델 로드
model = PPO.load("logs/best_model.zip", env=env)

# 🔧 하이퍼파라미터 강제 주입 (Fine-tuning)
NEW_LR = 3e-5  # 기존 3e-4 → 1/10로 감소
for param_group in model.policy.optimizer.param_groups:
    param_group["lr"] = NEW_LR

# 추가 학습
model.learn(total_timesteps=100_000)
model.save("logs/ppo_billiards_3cushion_finetuned")
```

---

### 3. Watching / Evaluation (Policy 테스트)

학습된 정책이 실제로 어떻게 치는지 시각적으로 확인:

```bash
python main.py --watch
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

> 현재 구현은 **환경 내부에서 직접 보상을 스케일링**하는 방식입니다.  
> (VecNormalize는 해제, 필요시 선택적으로 다시 사용할 수 있음)

- **목적**
  - 보상 스케일(예: 미스 -7, 성공 +50)의 차이가 너무 크면  
    Critic loss가 쉽게 폭발하거나 학습이 불안정해질 수 있음.
- **방법**
  - 환경 내부에서 `_calculate_reward()`로 **원시 보상(raw_reward)** 를 계산  
    (3쿠션 성공, 파울, 거리 개선 등)
  - `step()`에서:
    ```python
    raw_reward = self._calculate_reward(...)
    normalized_reward = raw_reward / self.reward_scale  # 예: reward_scale = 10.0
    return obs, normalized_reward, terminated, truncated, {"raw_reward": raw_reward}
    ```
  - PPO는 `normalized_reward`로 학습,  
    로그/분석 시에는 `info["raw_reward"]`를 사용해 실제 “당구 점수 감각” 유지.

- **장점**
  - 보상 분포를 명확히 컨트롤 가능.
  - SB3의 VecNormalize 내부 동작에 의존하지 않고,  
    논문/리포트에서 수식을 직접 명시하기 쉬움.

> 필요하다면 이후에 VecNormalize를 다시 켜서  
> **obs 정규화 + reward 추가 정규화**를 조합하는 것도 가능.

---

### 2. Stabilization Strategy

PPO 학습 안정성을 위한 전략들:

- **KL Divergence Monitoring**
  - `target_kl` 값을 기준으로, 한 업데이트에서 KL이 급격히 증가하면  
    → 해당 학습 루프를 조기 종료(Early Stopping)  
    → 정책이 갑자기 너무 많이 바뀌는 것을 방지

- **LR Scheduling**
  - 초기에는 비교적 큰 learning rate로 탐색(exploration)을 유도
  - 이후 학습이 진행될수록 `lr → decay`시키면서  
    **미세 조정(fine-tuning) 구간에서의 진동/발산**을 줄임
  - 예: `linear_schedule(initial_lr)` 형태의 스케줄 사용

---

## Research / Analysis 방향 (예시)

- **연속 득점 분포 분석**
  - 에피소드 당 연속 득점 수(run length) 히스토그램
  - 사람 선수의 평균 연속 득점과 비교

- **샷 선택 패턴 분석**
  - 템플릿 패턴(뒤돌, 앞돌, 옆돌, 빗겨, 대회전)별 성공률 / 선택률
  - 특정 배치에 대해서 사람이 선택하는 두께/당점과 모델의 행동 비교

- **Hyperparameter Study**
  - `γ`, `λ`, `clip_range`, `entropy_coef`, `lr` 변화에 따른
  - 수렴 속도 / 최종 성능 / policy collapse 여부 비교

---

## Troubleshooting Guide

### 1. PyGame 창이 안 뜨거나 바로 꺼짐

- 증상:
  - `render_mode="human"` 으로 실행했는데 창이 바로 꺼짐
- 확인 사항:
  - WSL(Windows Subsystem for Linux) 환경에서는 GUI가 바로 안 뜰 수 있음 → Native Windows Python 권장
  - `render_mode=None` 으로 학습, `render_mode="human"`은 관전 모드에서만 사용

### 2. GPU를 못 찾는 경우

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
    - 3쿠션 성공 시 보상이 실제로 양수로 들어오는지 (`raw_reward > 0`)  
    - miss 시에도 **너무 큰 음수**로 패널티를 주지 않았는지 확인
  - action space 확인:
    - `Box(low=-1, high=1, shape=(n,))` 형태에서  
      실제 물리 변수(힘, 각도, 스핀)로 변환하는 코드가 올바른지
  - 초기 템플릿:
    - 3쿠션이 물리적으로 가능한 배치인지 (불가능한 배치면 영원히 0점)

### 4. Policy Collapse (성능이 갑자기 급락)

- 증상:
  - 일정 단계까지 평균 reward가 오르다가  
    어느 순간부터 특정 행동에 고정되고, 성능이 급격히 떨어짐.
- 대응:
  - **학습률 낮추기**: 기존 `3e-4` → `3e-5` 또는 더 작게
  - **checkpoint 롤백**:
    - `logs/best_model.zip` 또는 이전 checkpoint를 로드
    - 위에서 설명한 Fine-tuning 방식으로 다시 학습
  - **entropy_coef 증가**:
    - 탐색(Exploration)을 조금 더 유지하도록 조정

---

## 📂 Repository Structure (예시)

```text
billiard-rl-agent/
├── environment_set.py     # BilliardEnv (Gym-style 환경)
├── simulation.py          # 물리 시뮬레이션 / Pymunk 관련 함수
├── play.py                # 학습 스크립트 (실험용)
├── utils/                 # 로깅, 시각화 유틸
├── logs/                  # 모델 / tensorboard / 모니터 로그
├── README.md
└── requirements.txt
```

---

## 📫 Contact

- Author: (이름 또는 GitHub ID)
- GitHub: [https://github.com/your-username/billiard-rl-agent](https://github.com/your-username/billiard-rl-agent)
- Issues / Pull Requests 환영
