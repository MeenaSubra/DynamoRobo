import time
from environment import RobotEnv
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer

def train():
    num_robots = 2
    env_width = 400
    env_height = 400
    static_obstacles = [(100, 100), (300, 100)]
    dynamic_obstacles = [
        {'position': [150, 150], 'velocity': [0.0, 0.0]},
        {'position': [250, 250], 'velocity': [0.0, 0.0]},
        {'position': [75, 225], 'velocity': [0.0, 0.0]},
        {'position': [325, 175], 'velocity': [0.0, 0.0]},
        {'position': [200, 50], 'velocity': [0.0, 0.0]}
    ]
    target_locations = [[350, 350], [50, 50]]

    env = RobotEnv(num_robots, static_obstacles, dynamic_obstacles, target_locations, env_width, env_height)
    state_dim = len(env._get_state()[0])
    action_dim = 2
    agents = [SACAgent(state_dim, action_dim) for _ in range(num_robots)]
    memory = ReplayBuffer(100000)

    num_episodes = 50000
    batch_size = 128
    print_interval = 500
    learn_start = 3000
    success_threshold = 5
    max_steps_per_episode = 500

    for episode in range(num_episodes):
        states = env.reset()
        episode_rewards = [0.0] * num_robots
        done = [False] * num_robots
        reached_target = [False] * num_robots
        step = 0

        while not all(done) and step < max_steps_per_episode:
            actions = [agents[i].act(states[i], explore=True) for i in range(num_robots)]
            next_states, rewards, current_done, _ = env.step(actions)
            env.update_dynamic_obstacles()

            for i in range(num_robots):
                memory.push(states[i], actions[i], rewards[i], next_states[i], current_done[i])
                episode_rewards[i] += rewards[i]
                dist_to_target = env._calculate_distance(env.robot_positions[i], env.target_locations[i])
                if not reached_target[i] and dist_to_target < success_threshold:
                    reached_target[i] = True
                    rewards[i] += 150

                done[i] = current_done[i]

            # Selective reset for robots that reached the target and waited
            reset_indices = [i for i, d in enumerate(done) if reached_target[i]]
            if reset_indices:
                print(f"Episode {episode}: Robots {reset_indices} reached target. Resetting them.")
                next_states_reset = env.reset(reset_indices)
                for i in reset_indices:
                    states[i] = next_states_reset[i]
                    done[i] = False
                    reached_target[i] = False

            states = next_states
            step += 1
            env.render()
            time.sleep(0.02)

            if all(not r for r in reached_target) and all(done): # End episode if all non-reached are done
                break

        avg_rewards = [total_reward / step if step > 0 else 0 for total_reward in episode_rewards]
        success_count = sum(reached_target)
        success_rate = success_count / num_robots * 100
        print(f"Episode {episode}: Avg Rewards = {avg_rewards}, Success Rate = {success_rate:.2f}%")

    print("Training finished. Starting evaluation...")
    evaluate(env, agents)
    env.close()

def evaluate(env, agents, num_eval_episodes=10, max_steps_per_episode=500):
    success_threshold = 5
    total_successes = 0
    for episode in range(num_eval_episodes):
        states = env.reset()
        done = [False] * env.num_robots
        reached_target = [False] * env.num_robots
        episode_success = [False] * env.num_robots
        step = 0
        while not all(done) and step < max_steps_per_episode:
            actions = [agents[i].act(states[i], explore=False) for i in range(env.num_robots)]
            next_states, _, current_done, _ = env.step(actions)
            env.update_dynamic_obstacles()
            states = next_states
            env.render()
            time.sleep(0.1)
            step += 1
            for i in range(env.num_robots):
                dist_to_target = env._calculate_distance(env.robot_positions[i], env.target_locations[i])
                if not reached_target[i] and dist_to_target < success_threshold:
                    reached_target[i] = True
                    reached_time = time.time()
                if reached_target[i] and time.time() - (reached_time if 'reached_time' in locals() else 0) >= 1.0:
                    done[i] = True
                    episode_success[i] = True
                else:
                    done[i] = current_done[i]

            if all(done):
                print(f"Evaluation Episode {episode + 1} finished in {step} steps.")
                break
        else:
            print(f"Evaluation Episode {episode + 1} reached max steps.")

        if all(episode_success):
            total_successes += 1

    accuracy = (total_successes / num_eval_episodes) * 100 if num_eval_episodes > 0 else 0
    print(f"\nEvaluation Accuracy: {accuracy:.2f}% (All robots reached target and waited in an episode)")
    print("Evaluation finished. Closing Pygame window.")

if __name__ == '__main__':
    train()