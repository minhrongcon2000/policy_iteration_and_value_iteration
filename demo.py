from environment.l2_environment import L2Environment
from algorithm.policy_iter import GridPolicyIteration
import seaborn as sns
import matplotlib.pyplot as plt

# initialize environment
env = L2Environment(10, traffic_p=0.25)

# solve environment using either policy iteration or value iteration
# details in algorithm package
solver = GridPolicyIteration(env=env, discounted_factor=0.9)
solver.solve_env()

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.heatmap(
    env.env_array, 
    cmap=sns.cm.rocket_r,
    linewidth=.5,
    ax=ax[0]
)
sns.heatmap(
    env.perform_policy(policy=solver.optimal_policy),
    cmap=sns.cm.rocket_r,
    linewidth=.5,
    ax=ax[1]
)
plt.show()