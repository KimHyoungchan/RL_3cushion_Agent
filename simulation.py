# watch.py
import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from environment_set import BilliardEnv

MODEL_NAME = "ppo_billiards_3cushion"
STATS_PATH = "vec_normalize.pkl"

# 1. 화면을 켜고 환경 생성
env = BilliardEnv(render_mode="human")
env = DummyVecEnv([lambda: env])

if os.path.exists(STATS_PATH):
    print(f"정규화 통계({STATS_PATH})를 로드합니다.")
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False     
    env.norm_reward = False  
else:
    print("정규화 통계 파일이 없습니다. (모델이 정상 작동하지 않을 수 있음)")

# 2. 저장된 모델 불러오기 
try:
    model = PPO.load("ppo_billiards_3cushion") 
    print("학습된 모델을 로드했습니다.")
except:
    model = None
    print("모델이 없어 랜덤 행동을 보여줍니다.")

# 3. 시뮬레이션 시작
obs = env.reset()
print("시뮬레이션 감상 시작 (종료하려면 창을 닫거나 Ctrl+C)")

for i in range(100): # 100번의 샷을 구경
    if model:
        action, _states = model.predict(obs, deterministic=False)
    else:
        action = env.action_space.sample() # 랜덤 행동
        
    obs, reward, terminated, info = env.step(action)
    
    # 샷 결과 출력
    print(f"Shot {i+1}: Reward {reward[0]:.2f}")
    
    if terminated:
        obs = env.reset()

env.close()