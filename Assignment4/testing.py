from hiive.mdptoolbox import mdp, example
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

if __name__ == "__main__":

    # Basic Example
    P, R = example.forest()
    vi = mdp.ValueIteration(P, R, 0.9)
    vi.run()

    random_map = generate_random_map(size=10, p=0.98)
    P, R = example.openai("FrozenLake-v1", desc=random_map)
    vi = mdp.ValueIteration(P, R, 0.9)
    vi.run()
    print(vi.policy)
