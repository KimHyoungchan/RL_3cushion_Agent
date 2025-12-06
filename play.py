import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from environment_set import BilliardEnv
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# [설정] True: 학습 / False: 관전
TRAIN_MODE = True

#MODEL_PATH = "sac_billiards_3cushion"
MODEL_PATH = "ppo_billiards_3cushion"
LOG_DIR = "./logs/" # 로그 저장 폴더
STATS_PATH = "vec_normalize.pkl"
# 로그 폴더 생성
os.makedirs(LOG_DIR, exist_ok=True)

# 학습 진행률(progress_remaining)에 따라 학습률을 선형적으로 조절 (1.0 -> 0.0)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if TRAIN_MODE:
    print("====================학습 시작====================")

    # 1. 환경 생성 및 Monitor 래퍼
    env = BilliardEnv(render_mode=None)
    env = Monitor(env, LOG_DIR) 
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]), # 용량 늘림 (64->128->256 / layer 수 2->3)
        log_std_init=0.0 
    )

    # 2. 모델 설정
    if os.path.exists(MODEL_PATH + ".zip"):
        print("====================기존 모델 로드====================")
        if os.path.exists(STATS_PATH):
            print("정규화 통계 로드 완료")
            env = VecNormalize.load(STATS_PATH, env.venv)
            env.training = True
            env.norm_reward = True
        
        custom_objects = {
        "learning_rate": 0.00003,
        "clip_range": lambda _: 0.1,
        }
        model = PPO.load(MODEL_PATH, env=env, custom_objects=custom_objects)
        #model = SAC.load(MODEL_PATH, env=env)
        
        NEW_LR = linear_schedule(0.0001)
        NEW_CLIP_RANGE = 0.15
        NEW_TARGET_KL = 0.1

        model.learning_rate = NEW_LR
        model.clip_range = lambda _: NEW_CLIP_RANGE
        model.target_kl = NEW_TARGET_KL

        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = NEW_LR
        
        print(f"하이퍼파라미터 수정 완료")
        print(f"New Learning Rate: {NEW_LR}")
        #print(f"New Clip Range: {NEW_CLIP_RANGE}")
        print(f"New Target kl: {NEW_TARGET_KL}")

    else:
        print("====================새 모델 생성====================")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, #1차 학습 : 0.0003 시작
            n_steps=2048, #2048->4096->2048->(4096(90만),2048())
            batch_size=256,
            gamma=0.99,
            ent_coef=0.03,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/",
            target_kl=0.1,
            n_epochs=10
            )
        
        # model = SAC(
        #     "MlpPolicy", 
        #     env, 
        #     verbose=1, 
            
        #     learning_rate=linear_schedule(0.0003),
        #     buffer_size=1000000, #100만 개의 과거 경험을 기억함 (메모리 약 4~8GB 사용)
        #     learning_starts=10000, # 처음 1만 스텝은 학습 안 하고 데이터만 모음 (Random Action)
        #     batch_size=256,      # 한 번에 256개의 과거 기억을 꺼내서 복습
        #     tau=0.005,           # 타겟 네트워크 업데이트 비율 (부드러운 업데이트)
        #     gamma=0.99,          # 미래 보상 할인율
        #     train_freq=1,        # 매 스텝마다 학습
        #     gradient_steps=1,    # 한 번 학습할 때 업데이트 횟수
        #     ent_coef='auto',     # [중요] 엔트로피(탐험)를 알아서 조절함
        #     policy_kwargs=policy_kwargs,
        #     tensorboard_log="./logs/"
        # )

    

    # 3. 학습 시작 (log_interval=10 -> 10번 업데이트마다 로그 출력)
    try:
        model.learn(total_timesteps=100000, log_interval=10)
    except KeyboardInterrupt:
        print("====================학습 강제 중단 (모델 저장)====================")

    # 4. 모델 저장
    model.save(MODEL_PATH)
    env.save(STATS_PATH)
    print("====================학습 완료 (모델 저장)====================")

    # 5. 학습 결과 시각화
    print("====================학습 결과 그래프 생성====================")
    try:
        # Monitor가 저장한 로그 파일(.monitor.csv) 읽기
        x, y = ts2xy(load_results(LOG_DIR), 'timesteps')
        
        # 이동 평균(Moving Average) 계산
        window = 50
        if len(y) > window:
            y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
            x_smooth = x[len(x) - len(y_smooth):]
        else:
            y_smooth, x_smooth = y, x

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, alpha=0.3, label='Raw Reward') # 실제 reward (흐리게)
        plt.plot(x_smooth, y_smooth, color='red', label='Moving Average') # 평균 reward 변화 (빨간선)
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title('Billiards RL Training Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show() # 그래프 창 띄우기
    except Exception as e:
        print(f"그래프 그리기 실패 (Exception): {e}")

else:
    # 실행 모드
    print("====================시뮬레이션 모드====================")
    env = BilliardEnv(render_mode="human")
    env = DummyVecEnv([lambda: env])
    if os.path.exists(STATS_PATH):
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False 
        env.norm_reward = False
    
    if os.path.exists(MODEL_PATH + ".zip"):
        model = PPO.load(MODEL_PATH)
        #model = SAC.load(MODEL_PATH) # SAC 로드
    else:
        model = None
        
    obs, _ = env.reset()
    for i in range(100):
        if model:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Shot {i+1} | Reward: {reward:.4f}")
        
        if terminated: 
            obs, _ = env.reset()

    env.close()
