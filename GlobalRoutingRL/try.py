import wandb
import os
from time import time, sleep
import torch
import numpy as np
from DTQN import set_global_seed

def __init__(self,  sess, gridgraph, render=False):
  self
  self.replay.is_burn_in = False
  self.gridgraph = gridgraph 

            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Use `--disable-wandb` to log locally.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time limit allowed for job. Useful for some cluster jobs such as slurm.",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=2_000_000,
        help="Number of steps to train the agent.",
    )
    parser.add_argument(
        "--tuf",
        type=int,
        default=10_000,
        help="How many steps between each (hard) target network update.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--buf-size",
        type=int,
        default=500_000,
        help="Number of timesteps to store in replay buffer. Note that we store the max length episodes given by the environment, so episodes that take longer will be padded at the end. This does not affect training but may affect the number of real observations in the buffer.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5_000,
        help="How many training timesteps between agent evaluations.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for each evaluation period.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Pytorch device to use."
    )
    parser.add_argument(
        "--context",
        type=int,
        default=50,
        help="For DRQN and DTQN, the context length to use to train the network.",
    )

    parser.add_argument(
        "--a-embed",
        type=int,
        default=0,
        help="The number of features to give each action. A value of 0 will prevent the policy from using the previous action.",
    )
    parser.add_argument(
        "--in-embed",
        type=int,
        default=128,
        help="The dimensionality of the network. In the transformer, this is referred to as `d_model`.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=-1,
        help="The maximum number of steps allowed in the environment. If `env` has a `max_episode_steps`, this will be inferred. Otherwise, this argument must be supplied.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed to use.")
    parser.add_argument(
        "--save-policy",
        action="store_true",
        help="Use this to save the policy so you can load it later for rendering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out evaluation results as they come in to the console.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enjoy mode (NOTE: must have a trained policy saved).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=50,
        help="This is how many (intermediate) Q-values we use to train for each context. To turn off intermediate Q-value prediction, set `--history 1`. To use the entire context, set history equal to the context length.",
    )
    # DTQN-Specific
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of heads to use for the transformer.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of transformer blocks to use for the transformer.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability."
    )
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--gate",
        type=str,
        default="res",
        choices=["res", "gru"],
        help="Combine step to use.",
    )


    parser.add_argument(
        "--bag-size", type=int, default=0, help="The size of the persistent memory bag."
    )
    parser.add_argument('--model_no',dest='model_file_no',type=str)     # 模型编号，类型为字符串
    # For slurm
    # parser.add_argument(
    #    "--slurm-job-id",
    #    default=0,
    #    type=str,
    #    help="The `$SLURM_JOB_ID` assigned to this job.",
    # )

    return parser.parse_args()


def evaluate(self,
    agent,
    eval_episodes=20,
    model_file=None, stat=False
):
  
    # Set networks to eval mode (turns off dropout, etc.)
    agent.eval_on()
    if model_file is not None:  # 如果提供了模型文件
        self.saver.restore(self.sess, model_file)  # 加载这个模型
         
    reward_list = []
    cum_reward = 0.0
    for episode in np.arange(eval_episodes):  # 对环境执行一定数量（no）的情节（episodes）
        agent.context_reset() # 重置上下文环境
        done = False
        episode_reward = 0.0
        state = self.gridgraph.reset()
        while not done:
            observation = self.gridgraph.state2obsv()
            q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
            action = self.greedy_policy(q_values)
            nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
            state = nextstate
            episode_reward = episode_reward+reward
            cum_reward = cum_reward+reward
        reward_list.append(episode_reward)
    agent.eval_off()    
    if stat:
        return cum_reward, reward_list
    else:
        return cum_reward
        # 在每个情节中，不断地获取当前状态的观察值，使用贪婪策略（greedy_policy）基于Q网络（qNetwork）的输出选择动作，并执行动作来获取奖励，直到情节终止。这一过程中累积的奖励用于评估模型的性能。
    
    # Set networks back to train mode
    agent.eval_off()

def train(self,twoPinNum,twoPinNumEachNet,netSort,savepath,model_file=None
    agent,
    eps: int,
    eval_frequency: int,
    eval_episodes: int,
    policy_path: str,
    verbose: bool = False,
) -> None:


    # 初始化了一些用于存储解决方案组合和奖励数据的列表
    self.max_episodes = 200 #20000 #200
    reward_log = []
    test_reward_log = []
    test_episode = []
    solution_combo = []
    reward_plot_combo = []
    reward_plot_combo_pure = []

    for episode in np.arange(self.max_episodes*len(self.gridgraph.twopin_combo)):
    # 对于每个可能的两终端网络，算法将尝试执行最多self.max_episodes次迭代，以训练模型或评估其性能。

        # n_node = len([n.name for n in tf.get_default_graph().as_graph_def().node])
        # print("No of nodes: ", n_node, "\n")

        # print('Route:',self.gridgraph.route)
        solution_combo.append(self.gridgraph.route)  # 将当前的路由结果添加到solution_combo列表中。

        state, reward_plot, is_best = self.gridgraph.reset()  # 重置环境到初始状态，并获取初始状态、奖励以及是否是最佳状态的标志。
        reward_plot_pure = reward_plot-self.gridgraph.posTwoPinNum*100  # 计算纯粹的奖励值，通过从原始奖励中减去一个基于两终端网络数量的固定值。
        # print('reward_plot-self.gridgraph.posTwoPinNum*100',reward_plot-self.gridgraph.posTwoPinNum*100)

        # 每当episode可被twoPinNum整除时，将当前奖励和纯粹奖励分别添加到reward_plot_combo和reward_plot_combo_pure列表中。
        if (episode) % twoPinNum == 0:
            reward_plot_combo.append(reward_plot)
            reward_plot_combo_pure.append(reward_plot_pure)
        is_terminal = False
        rewardi = 0.0
        if episode % 100 == 0:  # 每100个迭代更新一次网络参数。
            self.network_assign()
   


    start_time = time()
    # Turn on train mode
    agent.eval_off()
    # Choose an environment at the start and on every episode reset.
    agent.context_reset()
    
    for timestep in np.arange(self.max_episodes*len(self.gridgraph.twopin_combo)):
    # for timestep in range(agent.num_train_steps, total_steps):
        done = step(agent, eps)

        if done:
            agent.replay_buffer.flush()
            agent.context_reset(self.gridgraph.reset())
        agent.train()
        eps.anneal()

        if timestep % eval_frequency == 0:
            hours = (time() - start_time) / 3600
            # Log training values
            log_vals = {
                "losses/TD_Error": agent.td_errors.mean(),
                "losses/Grad_Norm": agent.grad_norms.mean(),
                "losses/Max_Q_Value": agent.qvalue_max.mean(),
                "losses/Mean_Q_Value": agent.qvalue_mean.mean(),
                "losses/Min_Q_Value": agent.qvalue_min.mean(),
                "losses/Max_Target_Value": agent.target_max.mean(),
                "losses/Mean_Target_Value": agent.target_mean.mean(),
                "losses/Min_Target_Value": agent.target_min.mean(),
                "losses/hours": hours,
            }
            # Perform an evaluation for each of the eval environments and add to our log
            sr, ret, length = evaluate(agent, eval_episodes)

            log_vals.update(
                {
                    f"SuccessRate": sr,
                    f"Return": ret,
                    f"EpisodeLength": length,
                }
            )

            # Commit the log values.
            logger.log(
                log_vals,
                step=timestep,
            )

            if verbose:
                print(
                    f" Success Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}, Hours: {hours:.2f}"
                )

        if save_policy and timestep % 50_000 == 0:
            torch.save(agent.policy_network.state_dict(), policy_path)

        if time_remaining and time() - start_time >= time_remaining:
            print(
                f"Reached time limit. Saving checkpoint with {agent.num_train_steps} steps completed."
            )

            agent.save_checkpoint(
                policy_path,
                wandb.run.id if logger == wandb else None,
                mean_success_rate,
                mean_reward,
                mean_episode_length,
                eps,
            )
            return



def step(agent, eps: float) -> bool:
    """Use the agent's policy to get the next action, take it, and then record the result.

    Arguments:
        agent:  the agent to use.
        eps:    the epsilon value (for epsilon-greedy policy)

    Returns:
        done: bool, whether or not the episode has finished.
    """
    action = agent.get_action(epsilon=eps.val)
    next_obs, reward, done, info = self.gridgraph.step(action)


def burn_in_memory_search(self,agent,observationCombo,actionCombo,rewardCombo,
    observation_nextCombo,is_terminalCombo): # Burn-in with search
    agent.context_reset()
    done = False
    print('Start burn in with search algorithm...')
    for i in range(len(observationCombo)):
        observation = observationCombo[i]
        action = actionCombo[i]
        reward = rewardCombo[i]
        observation_next = observation_nextCombo[i]
        buffer_done = is_terminalCombo[i]

    agent.observe(observation_next, action, reward, buffer_done)
    buffer_done = done

    agent.replay_buffer.flush()
    print('Burn in with search algorithm finished.')


def get_agent(
    model_str: str,
    embed_per_obs_dim: int,
    action_dim: int,
    inner_embed: int,
    buffer_size: int,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    context_len: int,
    max_env_steps: int,
    history: int,
    target_update_frequency: int,
    gamma: float,
    num_heads: int = 1,
    num_layers: int = 1,
    dropout: float = 0.0,
    identity: bool = False,
    gate: str = "res",
    pos: str = "learned",
    bag_size: int = 0,
):
    """Function to create the agent. This will also set up the policy and target networks that the agent needs.
    Arguments:
        model_str: str, the name of the Q-function model we are going to use.
        ember_per_obs_dim: int, the number of features to give each dimension of the observation. This is only used for discrete domains.
        action_dim: int, the number of features to give each action.
        inner_embed: int, the size of the main transformer model.
        buffer_size: int, the number of transitions to store in the replay buffer.
        device: torch.device, the device to use for training.
        learning_rate: float, the learning rate for the ADAM optimiser.
        batch_size: int, the batch size to use for training.
        context_len: int, the maximum sequence length to use as input to the network.
        max_env_steps: int, the maximum number of steps allowed in the environment before timeout. This will be inferred if not explicitly supplied.
        history: int, the number of Q-values to use during training for each sample.
        target_update_frequency: int, the number of training steps between (hard) target network update.
        gamma: float, the discount factor.
        -DTQN-Specific-
        num_heads: int, the number of heads to use in the MultiHeadAttention.
        num_layers: int, the number of transformer blocks to use.
        dropout: float, the dropout percentage to use.
        identity: bool, whether or not to use identity map reordering.
        gate: str, which combine step to use (residual skip connection or GRU)
        pos: str, which type of position encoding to use ("learned", "sin", or "none")
        bag_size: int, the size of the persistent memory bag

    Returns:
        the agent we created with all those arguments, complete with replay buffer, context, policy and target network.
    """
    # All envs must have the same observation shape
    env_obs_length = 12
    env_obs_mask = -1
    """if max_env_steps <= 0:
        max_env_steps = max([env_processing.get_env_max_steps(env) for env in envs])"""

   
    # Keep the history between 1 and context length
    if history < 1 or history > context_len:
        print(
            f"History must be 1 < history <= context_len, but history is {history} and context len is {context_len}. Clipping history to {np.clip(history, 1, context_len)}..."
        )
        history = np.clip(history, 1, context_len)
    # All envs must share same action space
    num_actions = 6

    # def wandb_init(config, group_keys, **kwargs) -> None:
    # wandb.init(
    #    project=config["project_name"],
    #   group="_".join(
    #       [f"{key}={val}" for key, val in config.items() if key in group_keys]
    #    ),
    #   config=config,
    #   **kwargs,
    # )

def run_experiment(args):
    """Uses the command-line arguments to create the agent and associated tools, then begin training."""
    start_time = time()
    # Create set seed, create RL agent
    device = torch.device(args.device)
    set_global_seed(args.seed, *(envs + eval_envs))

    eps = 1.0
    # eps = epsilon_anneal.LinearAnneal(1.0, 0.1, args.num_steps // 10)


    
    agent = get_agent(
        args.model,
        args.obs_embed,
        args.a_embed,
        args.in_embed,
        args.buf_size,
        device,
        args.lr,
        args.batch,
        args.context,
        args.max_episode_steps,
        args.history,
        args.tuf,
        args.discount,
        # DTQN specific
        args.heads,
        args.layers,
        args.dropout,
        args.identity,
        args.gate,
        args.pos,
        args.bag_size,
    )



    # Create logging dir
    policy_save_dir = os.path.join(
        os.getcwd(), "policies", args.project_name, *args.envs
    )
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(
        policy_save_dir,
        f"model={args.model}_envs={','.join(args.envs)}_obs_embed={args.obs_embed}_a_embed={args.a_embed}_in_embed={args.in_embed}_context={args.context}_heads={args.heads}_layers={args.layers}_"
        f"batch={args.batch}_gate={args.gate}_identity={args.identity}_history={args.history}_pos={args.pos}_bag={args.bag_size}_seed={args.seed}",
    )

    # Enjoy mode
    """if args.render:
        agent.policy_network.load_state_dict(
            torch.load(policy_path, map_location="cpu")
        )
        evaluate(agent, eval_envs[0], 1_000_000, render=True)"""

    # If there is already a saved checkpoint, load it and resume training if more steps are needed
    # Or exit early if we have already finished training.
    if os.path.exists(policy_path + "_mini_checkpoint.pt"):
        steps_completed = agent.load_mini_checkpoint(policy_path)["step"]
        print(
            f"Found a mini checkpoint that completed {steps_completed} training steps."
        )
        if steps_completed >= args.num_steps:
            print(f"Removing checkpoint and exiting...")
            if os.path.exists(policy_path + "_checkpoint.pt"):
                os.remove(policy_path + "_checkpoint.pt")
            exit(0)
        else:
            (
                wandb_id,
                mean_success_rate,
                mean_reward,
                mean_episode_length,
                eps_val,
            ) = agent.load_checkpoint(policy_path)
            eps.val = eps_val
            wandb_kwargs = {"resume": "must", "id": wandb_id}
    # Begin training from scratch
    else:
        wandb_kwargs = {"resume": None}
        # Prepopulate the replay buffer
        burn_in_memory_search(agent, 50_000)
        mean_success_rate = RunningAverage(10)
        mean_reward = RunningAverage(10)
        mean_episode_length = RunningAverage(10)

   # Logging setup
# logger = get_logger(policy_path, args, wandb_kwargs)"""

    time_remaining = (
        args.time_limit * 3600 - (time() - start_time) if args.time_limit else None
    )

    train(
        agent,
        args.envs,
        args.num_steps,
        eps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path,
        args.save_policy,
        logger,
        mean_success_rate,
        mean_reward,
        mean_episode_length,
        time_remaining,
        args.verbose,
    )

    """  # Save a small checkpoint if we finish training to let following runs know we are finished
        agent.save_mini_checkpoint(
            checkpoint_dir=policy_path, wandb_id=wandb.run.id if logger == wandb else None
        )"""



if __name__ == "__main__":
    run_experiment(get_args())
