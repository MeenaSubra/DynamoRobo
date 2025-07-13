import math
import random
import pygame
import time
import numpy as np

class RobotEnv:
    def __init__(self, num_robots, static_obstacles, dynamic_obstacles, target_locations, env_width, env_height):
        self.num_robots = num_robots
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.target_locations = target_locations
        self.env_width = env_width
        self.env_height = env_height

        self.robot_positions = [[random.uniform(0, env_width), random.uniform(0, env_height)] for _ in range(num_robots)]
        self.robot_velocities = [[0.0, 0.0]] * num_robots
        self.previous_distances = [float('inf')] * num_robots
        self.target_reached_time = [None] * num_robots
        self.out_of_bounds_count = [[0, 0]] * num_robots

        pygame.init()
        self.screen = pygame.display.set_mode((env_width, env_height))
        pygame.display.set_caption("Robot Navigation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)

    def reset(self, reset_indices=None):
        if reset_indices is None:
            self.robot_positions = [[random.uniform(0, self.env_width), random.uniform(0, self.env_height)] for _ in range(self.num_robots)]
            self.robot_velocities = [[0.0, 0.0]] * self.num_robots
            self.previous_distances = [float('inf')] * self.num_robots
            self.target_reached_time = [None] * self.num_robots
            self.out_of_bounds_count = [[0, 0]] * self.num_robots
        else:
            for i in reset_indices:
                self.robot_positions[i] = [random.uniform(0, self.env_width), random.uniform(0, self.env_height)]
                self.robot_velocities[i] = [0.0, 0.0]
                self.previous_distances[i] = float('inf')
                self.target_reached_time[i] = None
                self.out_of_bounds_count[i] = [0, 0]
        return self._get_state()

    def _calculate_relative_distances_to_obstacles(self, robot_index):
        distances = []
        for obstacle in self.static_obstacles:
            dist = np.linalg.norm(np.array(self.robot_positions[robot_index]) - np.array(obstacle))
            distances.append(dist)
        for obstacle in self.dynamic_obstacles:
            dist = np.linalg.norm(np.array(self.robot_positions[robot_index]) - np.array(obstacle['position']))
            distances.append(dist)
        return distances

    def _calculate_relative_distance_to_target(self, robot_index):
        return np.linalg.norm(np.array(self.target_locations[robot_index]) - np.array(self.robot_positions[robot_index]))

    def _calculate_relative_velocities_to_obstacles(self, robot_index):
        velocities = []
        for obstacle in self.dynamic_obstacles:
            relative_velocity = np.array(obstacle['velocity']) - np.array(self.robot_velocities[robot_index])
            velocities.append(relative_velocity.tolist())
        return velocities

    def _get_state(self):
        states = []
        max_obstacles = 10
        for i in range(self.num_robots):
            robot_state = []
            robot_state.extend(self.robot_positions[i])

            distances = self._calculate_relative_distances_to_obstacles(i)
            distances = distances[:max_obstacles]
            robot_state.extend(distances + [0.0] * (max_obstacles - len(distances)))

            robot_state.append(self._calculate_relative_distance_to_target(i))
            robot_state.extend(self.robot_velocities[i])

            velocities = self._calculate_relative_velocities_to_obstacles(i)
            velocities = velocities[:max_obstacles]
            robot_state.extend([item for sublist in velocities for item in sublist] + [0.0] * (max_obstacles * 2 - len([item for sublist in velocities for item in sublist])))

            robot_state = np.array(robot_state, dtype=np.float32)
            mean = np.mean(robot_state)
            std = np.std(robot_state)
            if std != 0:
                robot_state = (robot_state - mean) / std
            else:
                robot_state = (robot_state - mean)

            states.append(robot_state)
        return states

    def step(self, actions):
        next_positions = self._apply_actions(actions)
        rewards, dones = self._calculate_rewards(next_positions)
        self.robot_positions = next_positions
        next_states = self._get_state()
        return next_states, rewards, dones, {}

    def _apply_actions(self, actions):
        next_positions = []
        for i in range(self.num_robots):
            max_vel = 2.0
            dx = actions[i][0] * max_vel
            dy = actions[i][1] * max_vel

            new_x = max(0, min(self.env_width - 1, self.robot_positions[i][0] + dx))
            new_y = max(0, min(self.env_height - 1, self.robot_positions[i][1] + dy))
            next_positions.append([new_x, new_y])
            self.robot_velocities[i] = [dx, dy]
        return next_positions

    def _calculate_rewards(self, next_positions):
        rewards = []
        dones = []
        for i in range(self.num_robots):
            r1 = -0.001  # Even smaller negative reward for existing
            r2 = 0.0    # Reward for getting closer (more significant now)
            r3 = 0.0    # Penalty for collision
            r4 = 0.0    # Huge reward for reaching target
            r_obstacle = 0.0 # Small penalty for being close to an obstacle
            done = False

            current_distance = self._calculate_distance(next_positions[i], self.target_locations[i])

            # Very significant positive reward for getting closer
            distance_difference = self.previous_distances[i] - current_distance
            r2 += distance_difference * 2.0  # Increased multiplier

            # HUGE positive reward for reaching the target
            if current_distance < 5:
                r4 += 1000.0  # Significantly larger reward
                if self.target_reached_time[i] is None:
                    self.target_reached_time[i] = time.time()
                elif self.target_reached_time[i] is not None and time.time() - self.target_reached_time[i] >= 1.0:
                    done = True

            # Small negative reward for getting too close to obstacles (less impactful initially)
            for obstacle in self.static_obstacles:
                dist_to_obs = self._calculate_distance(next_positions[i], obstacle)
                if dist_to_obs < 15:
                    r_obstacle -= 0.05
            for obstacle in self.dynamic_obstacles:
                dist_to_obs = self._calculate_distance(next_positions[i], obstacle['position'])
                if dist_to_obs < 15:
                    r_obstacle -= 0.05

            # Moderate penalty for collision (to still discourage but not paralyze)
            if self._check_collision(next_positions[i], self.static_obstacles) or \
               self._check_collision(next_positions[i], [obs['position'] for obs in self.dynamic_obstacles]):
                r3 = -0.25
                done = True

            # Small penalty for going out of bounds (less impactful initially)
            if next_positions[i][0] <= 0 or next_positions[i][0] >= self.env_width - 1 or \
               next_positions[i][1] <= 0 or next_positions[i][1] >= self.env_height - 1:
                r4 -= 0.01
                self.out_of_bounds_count[i][0] += 1
            else:
                self.out_of_bounds_count[i][0] = 0

            if sum(self.out_of_bounds_count[i]) > 20: # Increased threshold
                done = True

            reward = r1 + r2 + r3 + r4 + r_obstacle
            rewards.append(reward)
            dones.append(done)
            self.previous_distances[i] = current_distance
        return rewards, dones

    def _check_collision(self, robot_pos, obstacles):
        for obstacle in obstacles:
            dist = self._calculate_distance(robot_pos, obstacle)
            if dist < 10:
                return True
        return False

    def _calculate_distance(self, pos1, pos2):
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

    def render(self):
        self.screen.fill((0, 0, 0))

        for target in self.target_locations:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(target[0]), int(target[1])), 8)

        for obstacle in self.static_obstacles:
            pygame.draw.rect(self.screen, (0, 0, 255), (obstacle[0] - 5, obstacle[1] - 5, 10, 10))

        for obstacle in self.dynamic_obstacles:
            pygame.draw.circle(self.screen, (255, 255, 255), (int(obstacle['position'][0]), int(obstacle['position'][1])), 7)

        for i, pos in enumerate(self.robot_positions):
            color = (0, 255, 0) if self.target_reached_time[i] is not None else (255, 0, 0)
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), 6)
            text_surface = self.font.render(str(i), True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(int(pos[0]), int(pos[1])))
            self.screen.blit(text_surface, text_rect)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

    def update_dynamic_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            speed = 0.8
            obstacle['position'][0] += random.uniform(-speed, speed)
            obstacle['position'][1] += random.uniform(-speed, speed)

            obstacle['position'][0] = max(0, min(self.env_width, obstacle['position'][0]))
            obstacle['position'][1] = max(0, min(self.env_height, obstacle['position'][1]))