# watch.py
import gymnasium as gym
from stable_baselines3 import PPO
from environment_set import BilliardEnv

# 1. 화면을 켜고 환경 생성
env = BilliardEnv(render_mode="human")

# 2. 저장된 모델 불러오기 
try:
    model = PPO.load("ppo_billiards_3cushion") 
    print("학습된 모델을 로드했습니다.")
except:
    model = None
    print("모델이 없어 랜덤 행동을 보여줍니다.")

# 3. 시뮬레이션 시작
obs, _ = env.reset()
print("시뮬레이션 감상 시작 (종료하려면 창을 닫거나 Ctrl+C)")

for i in range(100): # 10번의 샷을 구경
    if model:
        action, _states = model.predict(obs, deterministic=True)
    else:
        action = env.action_space.sample() # 랜덤 행동
        
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 샷 결과 출력
    print(f"Shot {i+1}: Reward {reward:.2f}")
    
    if terminated:
        obs, _ = env.reset()

env.close()