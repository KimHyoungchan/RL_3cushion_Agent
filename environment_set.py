import gymnasium as gym
from gymnasium import spaces
import pymunk
import pymunk.pygame_util
import numpy as np
import pygame
import math
#import random ## random으로 배치 고를때 사용했음

def get_segment_dist_sq(p, a, b):
    px, py = p.x, p.y
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    
    # 선분 AB 벡터
    ab_x, ab_y = bx - ax, by - ay
    # 벡터 AP
    ap_x, ap_y = px - ax, py - ay
    
    # AB 길이의 제곱
    ab_len_sq = ab_x**2 + ab_y**2
    
    if ab_len_sq == 0:
        return ap_x**2 + ap_y**2 # A와 B가 같은 점
        
    # 투영 비율 t 계산 (내적)
    t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
    
    # 선분 위로 제한 (0 ~ 1)
    t = max(0.0, min(1.0, t))
    
    # 선분 위의 가장 가까운 점 Q
    qx = ax + t * ab_x
    qy = ay + t * ab_y
    
    # P와 Q 사이의 거리 제곱
    return (px - qx)**2 + (py - qy)**2

class BilliardEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 120} 

    def __init__(self, render_mode=None):
        super(BilliardEnv, self).__init__()
        
        self.WIDTH = 800
        self.HEIGHT = 400
        self.BALL_RADIUS = 10 
        
        self.COLLISION_BALL = 1
        self.COLLISION_WALL = 2

        # Action Space: [-1 ~ 1] 로 정규화 (학습 최적화)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation Space: 6차원
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.space = None
        
        self.current_shot = 0
        self.max_shots = 30 # 연속 득점 최대 제한
        self.suc_cnt = 0 
        self.step_cnt = 0
        # --- 5개 기본 배치 ---
        self.templates = []
        
        # 1. 기본 배치 (초구)
        self.templates.append([[200, 300], [200, 200], [600, 200]])  #흰-노-빨
        
        # # 2. 뒤돌리기 패턴 (10개)
        # for i in range(10):
        #     self.templates.append([[150, 150 + i*10], [650, 300 - i*10], [150, 300]])
            
        # # 3. 제각돌리기 패턴 (10개)
        # for i in range(10):
        #     self.templates.append([[400, 100 + i*5], [400, 300 - i*5], [650, 200]])
            
        # # 4. 대회전 패턴 (10개)
        # for i in range(10):
        #     self.templates.append([[100, 100], [700, 100 + i*5], [700, 300]])
        
        # # 5. 옆돌리기 패턴 (10개)
        # for i in range(10):
        #     self.templates.append([
        #         [300 + i*20, 250],   # 수구 (중앙 아래쪽)
        #         [300 + i*20, 50],    # 1적구(노란공) (위쪽 쿠션 근처에 붙임)
        #         [700, 200 + i*10]    # 2적구() (반대편 코너 근처)
        #     ])

        # # 6. 빗겨치기 패턴 (10개)
        # for i in range(10):
        #     self.templates.append([
        #         [150 + i*10, 300],   # 수구
        #         [150 + i*10, 100],   # 1적구 (수구와 세로로 멀리 배치)
        #         [600, 150 + i*10]    # 2적구 (중앙 반대편)
        #     ])

        # # 7. 랜덤성 배치 (20개) - 나머지 채우기
        # for i in range(20):
        #      self.templates.append([
        #          [np.random.uniform(100, 700), np.random.uniform(100, 300)],
        #          [np.random.uniform(100, 700), np.random.uniform(100, 300)],
        #          [np.random.uniform(100, 700), np.random.uniform(100, 300)]
        #      ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.space = pymunk.Space()
        self.space.damping = 0.75 # 마찰계수 0.98(너무 오래 구름) -> 0.97 -> 0.95
        
        #환경 초기화
        self.cushion_count = 0
        self.hit_balls = set()
        self.cushion_count_at_second_hit = 0
        self.current_shot = 0
        self.first_hit_ball = None

        # 핸들러 등록
        h_wall = self.space.add_collision_handler(self.COLLISION_BALL, self.COLLISION_WALL)
        h_wall.begin = self._on_wall_hit
        
        h_ball = self.space.add_collision_handler(self.COLLISION_BALL, self.COLLISION_BALL)
        h_ball.begin = self._on_ball_hit

        self._create_walls()
        
        # 기본 배치 하나 선택 (1. 초구부터 시작 -> 에버 올라가면 랜덤배치 학습 시켜볼 예정)
        chosen_template = self.templates[0]
        
        # # 2. 노이즈 추가 함수
        # def get_pos(base_pos):
        #     nx = base_pos[0] + np.random.uniform(-5.0, 5.0) # +- 5픽셀 흔들기
        #     ny = base_pos[1] + np.random.uniform(-5.0, 5.0)
        #     # 화면 밖으로 안 나가게
        #     nx = np.clip(nx, 20, self.WIDTH - 20)
        #     ny = np.clip(ny, 20, self.HEIGHT - 20)
        #     return nx, ny
        
        cx, cy = chosen_template[0][0], chosen_template[0][1]
        o1x, o1y = chosen_template[1][0], chosen_template[1][1]
        o2x, o2y = chosen_template[2][0], chosen_template[2][1]

        self.cue_ball = self._create_ball(cx, cy, color=(255, 255, 255))
        self.obj_ball1 = self._create_ball(o1x, o1y, color=(255, 255, 0))
        self.obj_ball2 = self._create_ball(o2x, o2y, color=(255, 0, 0))
                
        self.start_pos = pymunk.Vec2d(self.cue_ball.position.x, self.cue_ball.position.y)
        self.min_red_dist = self.cue_ball.position.get_dist_sqrd(self.obj_ball2.position)
        self.min_yel_dist = self.cue_ball.position.get_dist_sqrd(self.obj_ball1.position)
        self.start_red_dist = self.min_red_dist
        self.start_yel_dist = self.min_yel_dist

        return self._get_obs(), {}

    def step(self, action):
        self.step_cnt += 1
        self.current_shot += 1 #현재 샷 횟수 
        raw_force, raw_angle = action

        # 기존 1500 -> 3000(너무 빠름) -> 2000(얘도 빠름) -> 1500 -> 1000 -> 800
        force_mag = (raw_force + 1) / 2 * 1000 

        # 각도: -pi ~ pi
        angle = raw_angle * math.pi 
        
        # Action 적용
        impulse_x = force_mag * math.cos(angle)
        impulse_y = force_mag * math.sin(angle)
        self.cue_ball.velocity += (impulse_x / self.cue_ball.mass, impulse_y / self.cue_ball.mass)
        
        # 상태 초기화
        self.cushion_count = 0
        self.hit_balls = set()
        self.cushion_count_at_second_hit = 0
        self.first_hit_ball = None
        
        # 1. 시작 시점의 거리 (기준값)
        self.min_red_dist = self.start_red_dist
        self.min_yel_dist = self.start_yel_dist

        # 시뮬레이션
        steps_simulated = 0
        while True:
            self.space.step(1/60.0)
            steps_simulated += 1

            if self.render_mode == "human":
                self.render()

            # 정지 확인
            total_vel = self.cue_ball.velocity.length + self.obj_ball1.velocity.length + self.obj_ball2.velocity.length
            # 성공 여부 확인
            is_jackpot = (len(self.hit_balls) >= 2 and self.cushion_count_at_second_hit >= 3)

            # 조건문 안에서 바로 기준(0.1 or 2.0)을 정해서 비교
            if total_vel < (0.01 if is_jackpot else 2.0) and steps_simulated > 10:
                break
            
            # 타임아웃도 마찬가지 (3000 or 800)
            if steps_simulated > (3000 if is_jackpot else 800):
                break
        
        # 보상 계산
        reward = self._calculate_reward(self.start_red_dist, self.min_red_dist, self.start_yel_dist, self.min_yel_dist)
        
        is_success = (reward >= 50.0)
        
        if is_success:
            terminated = False
            print(f"Get point! (point: {self.current_shot})")
            self.step_cnt -= 1
        else:
            terminated = True
            
        # 너무 많이 치면 강제 종료
        if self.current_shot >= self.max_shots:
            terminated = True
        
        return self._get_obs(), reward, terminated, False, {}

    def _calculate_reward(self, r_start_dist, r_end_dist, y_start_dist, y_end_dist):
        reward = -0.1 
        hit_count = len(self.hit_balls)
        
        improvement = 0.0

        if self.current_shot == 1:
            # 초구: 빨간공 기준 개선도
            imp = math.sqrt(r_start_dist) - math.sqrt(r_end_dist)
            improvement = imp
        else:
            # 초구 이후 : 둘 중 더 많이 가까워진 쪽
            imp_y = math.sqrt(y_start_dist) - math.sqrt(y_end_dist)
            imp_r = math.sqrt(r_start_dist) - math.sqrt(r_end_dist)
            improvement = max(imp_y, imp_r)
            
        # 거리 보상: 100픽셀 다가가면 1점
        distance_score = improvement * 0.01 
        
        #초구 노란공 치면 파울
        is_break_foul = False
        if self.current_shot == 1 and hit_count > 0:
            if self.first_hit_ball != self.obj_ball2:
                is_break_foul = True 
            elif self.first_hit_ball == self.obj_ball2:
                reward += 5

        if hit_count == 0:
            reward -= 7
            reward += distance_score * 2.0  # 거리 보상 강화
            #print(f"Can't hit any ball! (Cushions: {self.cushion_count})")

        if is_break_foul:
            reward -= 10.0 # 파울 페널티

        elif hit_count == 1:
            reward += 3.0 
            reward += self.cushion_count * 1.0
            #print(f"hit just one ball! (Cushions: {self.cushion_count})")
            
        elif hit_count == 2:
            valid_cushions = self.cushion_count_at_second_hit
            
            if valid_cushions >= 3:
                reward += 50.0
                self.suc_cnt += 1 
                if self.current_shot > 1:
                    reward += 10
                print(f" 3-Cushion : Get Point! (Cushions: {self.cushion_count} | cnt : {self.suc_cnt} | avg. : {self.suc_cnt/self.step_cnt}, steps : {self.step_cnt}, reward : {reward})")
            else:
                reward += 7.0
                reward += valid_cushions * 3.0
                #print(f" hit 2 balls before satisfing 3 cushions! (Cushions: {self.cushion_count})")
                
        return reward
    
    #쿠션 수 count
    def _on_wall_hit(self, arbiter, space, data):
        ball_shape = arbiter.shapes[0]
        if ball_shape.body == self.cue_ball:
            if self.cushion_count == 0:
                hit_pos = self.cue_ball.position
                seg_dist = get_segment_dist_sq(self.obj_ball2.position, self.start_pos, hit_pos)
                self.min_red_dist = seg_dist
            if (self.cushion_count < 3): #3 이상은 count x
                self.cushion_count += 1
        return True
    
    #공 타격 수 count
    def _on_ball_hit(self, arbiter, space, data):
        shape_a, shape_b = arbiter.shapes
        body_a, body_b = shape_a.body, shape_b.body
        target = None
        if body_a == self.cue_ball: target = body_b
        elif body_b == self.cue_ball: target = body_a
        if target:
            if self.first_hit_ball is None:
                self.first_hit_ball = target
                
            self.hit_balls.add(id(target))
            if len(self.hit_balls) == 2 and self.cushion_count_at_second_hit == 0:
                self.cushion_count_at_second_hit = self.cushion_count
        return True

    #
    def _get_obs(self):
        obs = np.array([
            self.cue_ball.position.x / self.WIDTH, self.cue_ball.position.y / self.HEIGHT,
            self.obj_ball1.position.x / self.WIDTH, self.obj_ball1.position.y / self.HEIGHT,
            self.obj_ball2.position.x / self.WIDTH, self.obj_ball2.position.y / self.HEIGHT,
        ], dtype=np.float32)
        return obs

    #
    def _create_ball(self, x, y, color):
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, self.BALL_RADIUS)
        body = pymunk.Body(mass, moment)
        body.position = x, y
        shape = pymunk.Circle(body, self.BALL_RADIUS)
        shape.elasticity = 0.95
        shape.friction = 0.5
        shape.color = color
        shape.collision_type = self.COLLISION_BALL
        self.space.add(body, shape)
        return body
    
    #
    def _create_walls(self):
        rects = [
            [(self.WIDTH/2, -10), (self.WIDTH, 20)],
            [(self.WIDTH/2, self.HEIGHT+10), (self.WIDTH, 20)],
            [(-10, self.HEIGHT/2), (20, self.HEIGHT)],
            [(self.WIDTH+10, self.HEIGHT/2), (20, self.HEIGHT)]
        ]
        for pos, size in rects:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = pos
            shape = pymunk.Poly.create_box(body, size)
            shape.elasticity = 0.7
            shape.collision_type = self.COLLISION_WALL
            self.space.add(body, shape)

    #
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()
            
        self.screen.fill((0, 80, 0)) 
        
        balls = [(self.cue_ball, (255,255,255)), (self.obj_ball1, (255,255,0)), (self.obj_ball2, (255,0,0))]
        for body, color in balls:
            pos = (int(body.position.x), int(body.position.y))
            pygame.draw.circle(self.screen, color, pos, self.BALL_RADIUS)

        font = pygame.font.SysFont("Arial", 18)
        info_text = f"Shot: {self.current_shot}/{self.max_shots} | C: {self.cushion_count} | H: {len(self.hit_balls)}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        
        # FPS를 높여서 화면을 더 빠르게(0 : 제한 없음, 120 : 2배속)
        self.clock.tick(120) 

    def close(self):
        if self.screen is not None:
            pygame.quit()
