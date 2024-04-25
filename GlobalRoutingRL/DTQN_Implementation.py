# from __future__ import annotations
from typing import Tuple, Optional, Union
from enum import Enum
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import keras, tensorflow as tf, numpy as np, sys, copy, argparse, random
import matplotlib.pyplot as plt
import math
import os

d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
# d_k = d_v = 64  # K(=Q), V的维度
num_heads = 8     # Multi-Head Attention设置为8
num_layers = 6
vocab_sizes = 10
obs_dim = 12
batch = 32
embed_size = 16
num_actions = 6
# action_embed_dim = 128
dropout = 0.1
buffer_size = 100
"""learning_rate = 0.0003"""

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.in_proj_bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    
class ObservationEmbeddingRepresentation(nn.Module):
    def __init__(self,  embed_dim: int):
        super().__init__()
        # 将每个离散观察映射到一个密集向量
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_sizes, embed_size),
            nn.Flatten(start_dim=-2),
            nn.Linear(embed_size * obs_dim, embed_dim),
        )

    def forward(self, obs: torch.Tensor):
        # batch, obs_dim= obs.size(0), obs.size(1)
        # Flatten batch and seq dims
        # obs = torch.flatten(obs, start_dim=0, end_dim=1)
        # Apply the linear transformation
        obs_embed = self.embedding(obs)
        # Reshape to the original batch and sequence dimensions
        obs_embed = obs_embed.reshape(batch, obs_dim, obs_embed.size(-1))
        return obs_embed


class ActionEmbeddingRepresentation(nn.Module):
    def __init__(self,  action_dim: int): 
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_actions, action_dim),
            nn.Flatten(start_dim=-2),
        )
    def forward(self, action: torch.Tensor):
        return self.embedding(action)


class PositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        position = torch.arange(obs_dim).unsqueeze(1) # 创建位置索引创建了一个从0到obs_dim-1的一维张量，并通过unsqueeze(1)方法将其变形为二维张量，用于表示序列中每个位置的索引。
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        ) # 计算的是位置编码的分母项，仅对偶数维度进行计算。这个分母项是按照位置编码的定义进行计算的，使用指数和对数函数来得到一组下降的数列。
        pos_encoding = torch.zeros(1, obs_dim, d_model) # 初始化一个全零的张量，用于存储最终的位置编码。与模型输入的批次尺寸保持一致。
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term) # 对于偶数维度（使用 0::2 切片选取），使用正弦函数计算位置编码；对于奇数维度（使用 1::2 切片选取），使用余弦函数计算。
        self.register_buffer('pos_encoding', pos_encoding) # 将位置编码注册为模型的一个持久状态，但不需要计算梯度

    def forward(self, obs_embed):
        """
        Args:
            obs_embed: 输入张量，其形状应为 [batch, obs_dim, d_model]
        Returns:
            返回添加了位置编码的输入张量。
        """
        # 返回位置编码与输入x相加的结果
        return obs_embed + self.pos_encoding[:, :obs_embed.size(1), :]

    
class ResGate(nn.Module):
    """Residual skip connection"""
 # ResGate 类实现了一个简单的残差跳跃连接，它允许信息直接跳过一些层，从而有助于减少深层网络中的梯度消失问题。
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y
    # forward 方法简单地将两个输入信号 x 和 y 相加并返回。这允许网络学习在保持输入信息的同时对输入信号进行适度调整的恒等变换，从而在增加网络深度时保持信息流。
    
class Transformer(nn.Module):
    def __init__(
        self,
        dropout: float, # Dropout 比例，用于防止过拟合。
        attn_gate, # 注意力子模块之后的组合层。
        mlp_gate, #前馈子模块之后的组合层。
    ):

        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        # 层归一化（LayerNorm）：在注意力机制和前馈网络（FFN）之前应用层归一化，有助于稳定训练过程。

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True, # 指定输入张量的第一个维度是否是批次大小。这意味着输入张量的形状应为 [batch_size, sequence_length, embed_dim]。
        )
        # 多头注意力（MultiheadAttention）：实现了 Transformer 的核心机制，即自注意力，允许模型在每个位置同时关注序列中的多个位置。
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        # 前馈网络（FFN）：由两个线性层和一个 ReLU 激活函数组成，其中间有一个扩展的隐藏层，用于进一步处理经过注意力机制的信息。
        self.attn_gate = attn_gate
        self.mlp_gate = mlp_gate # 门控机制（attn_gate 和 mlp_gate）：提供额外的灵活性，允许在注意力和 FFN 模块之后应用自定义的组合逻辑。
        # Just storage for attention weights for visualization
        self.alpha = None # 注意力权重（self.alpha）：存储注意力权重，可以用于后续的可视化分析。

        # Set up causal masking for attention
        self.attn_mask = nn.Parameter(
            torch.triu(torch.ones(obs_dim, obs_dim), diagonal=1),
            requires_grad=False,
        )
        self.attn_mask[self.attn_mask.bool()] = -float("inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多头自注意力（Multihead Attention）
        attention, self.alpha = self.attention(  
            x,
            x,
            x,
            attn_mask=self.attn_mask[: x.size(1), : x.size(1)], #
            average_attn_weights=True,  
        ) 
        x = self.attn_gate(x, F.relu(attention)) 
        x = self.layernorm1(x) 
        ffn = self.ffn(x)
        x = self.mlp_gate(x, F.relu(ffn)) 
        x = self.layernorm2(x) 
        return x


class DTQN(nn.Module):
    def __init__(self,**kwargs):
        self.action_embed_dim= kwargs.get('action_embed_dim', 128)
        self.attn_gate = kwargs.get('attn_gate',ResGate())
        self.mlp_gate = kwargs.get('mlp_gate', ResGate())


        self.obs_embed_dim = d_model - self.action_embed_dim   
        self.action_embedding = ActionEmbeddingRepresentation( # 创建一个动作嵌入
                action_dim=self.action_embed_dim
            )
        self.obs_embedding = (ObservationEmbeddingRepresentation( embed_dim=self.obs_embed_dim))
        self.position_encoding = PositionEncoding()

        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.Sequential( # 用 Python 列表推导式，创建了一个序列
            *[
                Transformer(
                    dropout, # 丢弃率
                    self.attn_gate, # 门控机制
                    self.mlp_gate,
                )
                for _ in range(num_layers)
            ]
        )

        self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, num_actions),
            )
        
        self.apply(init_weights)

    def forward(
        self,
        obss: torch.Tensor, # 序列观察数据
        actions: Optional[torch.Tensor] = None, # 可选的）动作
    ) -> torch.Tensor:
        # Embedding
        obs_embedding = self.obs_embedding(obss) # 对观察数据进行嵌入编码。
        action_embedding = self.action_embedding(actions) # 对动作进行嵌入，并将其与观察嵌入合并。
        action_embedding = torch.roll(action_embedding, 1, 1) # 动作嵌入向前滚动一位，以反映每个观察对应的先前动作。
        action_embedding[:, 0, :] = 0.0
        token_embedding = torch.concat([action_embedding, obs_embedding], dim=-1)

        #Encoding
         # [batch x obs_dim x d_model] -> [batch x obs_dim x d_model]
        working_memory = self.transformer_layers(
            self.dropout(
                token_embedding + self.position_encoding()[:, :obs_dim, :]
            )
        )  #  处理合并后的嵌入，同时加上位置编码，通过 Dropout 层进行正则化。
         
        output = self.ffn(working_memory) # 如果没有背包机制，直接将工作记忆通过 self.ffn 生成最终输出。

        return output[:, -obs_dim:, :]
    
class Context:    
    def __init__(
        self,
        obs_mask: int = 0,
        init_hidden: Tuple[torch.Tensor] = None, # 可选）RNNs使用的初始隐藏状态。
    ):
        self.timestep = 0
        self.obs_mask = obs_mask
        self.init_hidden = init_hidden

    def reset(self, obs: np.ndarray):
            """Resets to a fresh context"""
            self.obs = np.full([batch, obs_dim],self.obs_mask)
            # Initial observation
            self.obs[0] = obs

            self.action = np.random.randint(0, num_actions-1, size=(batch, 1))
            self.reward = np.full_like(self.action, 0)
            self.done = np.full_like(self.reward, 0, dtype=np.int32)
            self.hidden = self.init_hidden
            self.timestep = 0 # reset方法：重置上下文到初始状态，用于开始新的一集或一次交互。
    
    def add_transition(
        self, o: np.ndarray, a: int, r: float, done: bool
    ) -> Tuple[Union[np.ndarray, None], Union[int, None]]:
        """Add an entire transition. If the context is full, evict the oldest transition"""
        self.timestep += 1
        if self.timestep >= batch:
            self.obs = np.roll(self.obs, -1, axis=0)
            self.action = np.roll(self.action, -1, axis=0)
            self.reward = np.roll(self.reward, -1, axis=0)
            self.done = np.roll(self.done, -1, axis=0)


        t = min(self.timestep, batch - 1)

        self.obs[t] = o
        self.action[t] = np.array([a])
        self.reward[t] = np.array([r])
        self.done[t] = np.array([done])

        return # export方法：导出上下文中的数据。
  
    # add_transition方法：添加一个完整的转移（观测、动作、奖励、完成标志）。


"""     self.memory = []
        self.is_burn_in = False
        self.burn_in = burn_in """

class Replay_Memory:
    def __init__(
        self,
        obs_mask: int,
        # max_episode_length: int,
    ):
        # self.max_episode_length = max_episode_length
        self.obs_mask = obs_mask
        self.pos = [0, 0]

        self.obss = np.full(
                    [
                        buffer_size,
                        batch + 1,  # Keeps first and last obs together for +1
                        obs_dim,
                    ],
                    obs_mask,
                    dtype=np.float32,
        )
        
            # Need the +1 so we have space to roll for the first observation
        self.actions = np.zeros(
            [buffer_size, batch + 1, 1],
            dtype=np.uint8,
        )
        self.rewards = np.zeros(
            [buffer_size, batch, 1],
            dtype=np.float32,
        )
        self.dones = np.ones(
            [buffer_size, batch, 1],
            dtype=np.bool_,
        )
        self.episode_lengths = np.zeros([buffer_size], dtype=np.uint8)

    def store( # 存储一个时间步的观测、动作、奖励和结束信号。
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray, # 表明当前时间步是否结束了episode。
        episode_length: Optional[int] = 0, # 可选）当前episode的长度。

    ) -> None:
        episode_idx = self.pos[0] % buffer_size # 根据当前位置（self.pos）计算当前episode的索引（episode_idx）和在该episode中的观测索引（obs_idx）。
        obs_idx = self.pos[1]
        self.obss[episode_idx, obs_idx ] = obs
        self.next_obss[episode_idx, obs_idx + 1] = next_obs
        self.actions[episode_idx, obs_idx] = action
        self.rewards[episode_idx, obs_idx] = reward
        self.dones[episode_idx, obs_idx] = done # 在相应的存储结构中更新观测、动作、奖励和结束信号。
        self.episode_lengths[episode_idx] = episode_length # 更新当前episode的长度信息。
        self.pos = [self.pos[0], self.pos[1] + 1] # 更新位置索引，准备存储下一个时间步的信息。

    
    def sample( # 从缓冲区中随机抽样一批数据
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Exclude the current episode we're in
        valid_episodes = [
            i
            for i in range(min(self.pos[0], buffer_size))
            if i != self.pos[0] % buffer_size
        ]  # 从已存储的episode中选择有效的索引集合，排除当前正在进行的episode。
        episode_idxes = np.array(
            [[random.choice(valid_episodes)] for _ in range(batch_size)]
        ) # 随机选择指定数量的episode索引。
        transition_starts = np.array(
            [
                random.randint(
                    0, max(0, self.episode_lengths[idx[0]] - self.obs_dim)
                )
                for idx in episode_idxes
            ]
        ) # 对于每个选中的episode，随机选择一个起始点，确保选中的片段不会超出该episode的实际长度。
        transitions = np.array(
            [range(start, start + batch) for start in transition_starts]
        ) # 根据起始点和观察大小确定要抽样的转移区间。
        return (
            self.obss[episode_idxes, transitions],
            self.actions[episode_idxes, transitions],
            self.rewards[episode_idxes, transitions],
            self.obss[episode_idxes, 1 + transitions],
            self.actions[episode_idxes, 1 + transitions],
            self.dones[episode_idxes, transitions],
            np.clip(self.episode_lengths[episode_idxes], 0, self.context_len),
        ) # 根据计算出的索引和区间，从缓冲区中抽取相应的观测、动作、奖励和结束标志数据，以及下一步的观测和动作数据。

    
class Agent():
    def __init__(self, environment_name,sess,  gridgraph, render=False):
        self.epsilon = 0.05

        if environment_name == 'grid':
            self.gamma = 0.95
            self.max_episodes = 200 #20000 #200
            self.batch_size = 32
            self.render = render
            self.replay = Replay_Memory()
            self.gridgraph = gridgraph
            self.init = tf.global_variables_initializer()
            self.sess = sess
            # tf.summary.FileWriter("logs/", self.sess.graph)
            self.sess.run(self.init)
            self.saver = tf.train.Saver(max_to_keep=20)  # 保存和恢复模型的参数
            self.device = torch.device
            self.obs_tensor_type = torch.long

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        rnd = np.random.rand()  # 生成一个介于0到1之间的随机浮点数
        if rnd <= self.epsilon:
            return np.random.randint(num_actions)
        else:
            return torch.argmax(q_values[:, -1, :]).item()
        

    def train(self,twoPinNum,twoPinNumEachNet,netSort,savepath,model_file=None):
        if model_file is not None:
            self.saver.restore(self.sess, model_file)  # 提供了model_file参数，则使用TensorFlow的Saver对象恢复之前保存的模型，以便进行测试或进一步训练

        # 记录训练过程中的奖励信息和测试奖励信息，以及记录测试情节的信息。
        reward_log = []
        test_reward_log = []
        test_episode = []
        # if not self.replay.is_burn_in:
        # 	self.burn_in_memory()
        solution_combo = []

        # 初始化了一些用于存储解决方案组合和奖励数据的列表
        reward_plot_combo = []
        reward_plot_combo_pure = []
        a = 0
        r = 0
       
        for episode in np.arange(self.max_episodes*len(self.gridgraph.twopin_combo)):
        # 对于每个可能的两终端网络，算法将尝试执行最多self.max_episodes次迭代，以训练模型或评估其性能。
            solution_combo.append(self.gridgraph.route)  # 将当前的路由结果添加到solution_combo列表中。
        
            state, reward_plot, is_best = self.gridgraph.reset()  # 重置环境到初始状态，并获取初始状态、奖励以及是否是最佳状态的标志。
            reward_plot_pure = reward_plot-self.gridgraph.posTwoPinNum*100  # 计算纯粹的奖励值，通过从原始奖励中减去一个基于两终端网络数量的固定值。
            # print('reward_plot-self.gridgraph.posTwoPinNum*100',reward_plot-self.gridgraph.posTwoPinNum*100)

            if (episode) % twoPinNum == 0:
                reward_plot_combo.append(reward_plot)
                reward_plot_combo_pure.append(reward_plot_pure)
            is_terminal = False
            rewardi = 0.0
            Context.reset()
            rewardfortwopin = 0
            while not is_terminal:
                observation = self.gridgraph.state2obsv()  # 获取当前环境状态的观察值

            Context.add_transition(o=observation,done=is_terminal)
            """初始a,r怎么选取"""

            context_obs_tensor = torch.as_tensor( # 通过 torch.as_tensor 创建张量
                self.context.obs[: min(batch, self.context.timestep + 1)], # 截取了从开始到当前时间步（或最大存储长度，取决于哪个更小）的观察数据。
                dtype=self.obs_tensor_type, # 张量中元素的数据类型
                device=self.device, # 张量应该被存储在哪个设备上
            ).unsqueeze(0) # 通过 unsqueeze(0) 增加一个批处理维度,在第 0 个维度上增加一个维度
            context_action_tensor = torch.as_tensor(
                self.context.action[
                    : min(batch, self.context.timestep + 1)
                ],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            q_values = DTQN(obss=context_obs_tensor, actions=context_action_tensor)
            action = self.epsilon_greedy_policy(q_values)  # 根据epsilon-贪婪策略选择一个动作
			# print(action)
            nextstate, reward, is_terminal, debug = self.gridgraph.step(action)  # 执行选定的动作，环境返回下一个状态、本次动作的奖励、是否终止的标志和调试信息。
			# print(nextstate)
            observation_next = self.gridgraph.state2obsv()  # 生成下一个状态的观察值。
            Context.add_transition(o=observation,a=action,r=reward,done=is_terminal)
            self.replay.store([observation, action, reward, observation_next, is_terminal,self.context.timestep])
            state = nextstate
            rewardi = rewardi+reward
            rewardfortwopin = rewardfortwopin + reward  # 更新当前状态和累积的奖励。

            (
                obss,
                actions,
                rewards,
                next_obss,
                next_actions,
                dones,
                episode_lengths,
            ) = self.replay.sample(self.batch_size) # 如果没有配置背包，则仅从经验回放缓冲区抽取基本的转换数据。

            # 将抽取的数据（观察、动作、奖励、下一个观察、下一个动作、完成标志和背包中的观察与动作）转换为适当的PyTorch张量，并传输到指定的设备（CPU或GPU）上。
            # Obss and Next obss: [batch-size x hist-len x obs-dim]
            obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
            next_obss = torch.as_tensor(
                next_obss, dtype=self.obs_tensor_type, device=self.device
            )
            # Actions: [batch-size x hist-len x 1]
            actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            next_actions = torch.as_tensor(
                next_actions, dtype=torch.long, device=self.device
            )
            # Rewards: [batch-size x hist-len x 1]
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            # Dones: [batch-size x hist-len x 1]
            dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)

            # 使用策略网络（self.policy_network）对当前观察、动作以及（如果有的话）背包中的观察和动作进行处理，以得到每个动作的Q值。
            # obss is [batch-size x hist-len x obs-len]
            # then q_values is [batch-size x hist-len x n-actions]
            q_values = DTQN(obss=obss, actions=actions)

            with torch.no_grad():
                    argmax = torch.argmax(
                        DTQN(obss=next_obss, actions=next_actions),
                        dim=2,
                    ).unsqueeze(-1)
                    next_obs_q_values = self.target_network(
                    next_obss, next_actions)
                    next_obs_q_values = next_obs_q_values.gather(2, argmax).squeeze()

                    # 使用贝尔曼方程来计算目标 Q 值（targets），其中考虑了奖励、是否结束（dones）的标志，以及折扣因子（self.gamma）。
                    targets = rewards.squeeze() + (1 - dones.squeeze()) * (
                        next_obs_q_values * self.gamma
                    )
            q_values = q_values[:, -batch :]
            targets = targets[:, -batch :]
            # Calculate loss
            # 将当前 Q 值和目标 Q 值对齐到相同的历史长度，然后使用均方误差（MSE）损失函数来计算预测 Q 值和目标 Q 值之间的差距。
            loss = F.mse_loss(q_values, targets)
            # Optimization step
            self.optimizer.zero_grad(set_to_none=True) # 首先清除之前的梯度。
            loss.backward() # 计算损失关于网络参数的梯度。
            norm = torch.nn.utils.clip_grad_norm_( # 使用梯度裁剪（torch.nn.utils.clip_grad_norm_）防止梯度爆炸。
                DTQN.parameters(),
                self.grad_norm_clip,
                error_if_nonfinite=True,
            )
            # Logging
            self.grad_norms.add(norm.item())

            self.optimizer.step() # 执行一步梯度下降（self.optimizer.step()）来更新策略网络的参数。
            self.num_train_steps += 1

            if self.num_train_steps % self.target_update_frequency == 0:
                self.target_update() # 每隔一定训练步数（self.target_update_frequency），更新目标网络的参数以匹配策略网络的参数，这有助于稳定训练过程。
            
            reward_log.append(rewardi)
            self.gridgraph.instantrewardcombo.append(rewardfortwopin)

        score = self.gridgraph.best_reward	# 从gridgraph对象中获取最佳奖励best_reward并存储
        solution = self.gridgraph.best_route[-twoPinNum:]  # 提取与最佳路由相关的最后twoPinNum数量的布线解决方案并存储
        
        solutionDRL = []  # 为每个netSort中的网络初始化一个空列表solutionDRL

        for i in range(len(netSort)):
            solutionDRL.append([])

        print('twoPinNum',twoPinNum)
        print('solution',solution)

        # 组织并分配路由解决方案到不同的网络中。
        if self.gridgraph.posTwoPinNum  == twoPinNum:  # 当gridgraph对象中记录的两终端网络数posTwoPinNum等于twoPinNum时
            dumpPointer = 0
            for i in range(len(netSort)):  # 遍历netSort数组（代表网络排序）
                netToDump = netSort[i]
                for j in range(twoPinNumEachNet[netToDump]):  # 遍历twoPinNumEachNet（每个网络的两终端数）
                    # for k in range(len(solution[dumpPointer])):
                    solutionDRL[netToDump].append(solution[dumpPointer])  # 将最佳布线解决方案solution分配到对应的网络中
                    dumpPointer = dumpPointer + 1  # 追踪solution数组中当前考虑的元素。
        # print('best reward: ', score)
        # print('solutionDRL: ',solutionDRL,'\n')
        else:
            solutionDRL = solution  # 如果不满足条件，则直接将solution赋值给solutionDRL

        return solutionDRL,reward_plot_combo,reward_plot_combo_pure,solution,self.gridgraph.posTwoPinNum
        

    def test(self, model_file=None, no=20, stat=False):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		# uncomment this line below for videos
		# self.env = gym.wrappers.Monitor(self.env, "recordings", video_callable=lambda episode_id: True)
        if model_file is not None:  # 如果提供了模型文件
            self.saver.restore(self.sess, model_file)  # 加载这个模型
        reward_list = []
        cum_reward = 0.0
        for episode in np.arange(no):  # 对环境执行一定数量（no）的情节（episodes）
            episode_reward = 0.0
            state = self.gridgraph.reset()
            is_terminal = False
            while not is_terminal:
                observation = self.gridgraph.state2obsv()

                Context.add_transition(o=observation,a=action,r=reward,done=is_terminal) 
                context_obs_tensor = torch.as_tensor( # 通过 torch.as_tensor 创建张量
                self.context.obs[: min(batch, self.context.timestep + 1)], # 截取了从开始到当前时间步（或最大存储长度，取决于哪个更小）的观察数据。
                dtype=self.obs_tensor_type, # 张量中元素的数据类型
                device=self.device, # 张量应该被存储在哪个设备上
                ).unsqueeze(0) # 通过 unsqueeze(0) 增加一个批处理维度,在第 0 个维度上增加一个维度
                context_action_tensor = torch.as_tensor(
                    self.context.action[
                        : min(batch, self.context.timestep + 1)
                    ],
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                q_values = DTQN(obss=context_obs_tensor, actions=context_action_tensor)

                action = self.greedy_policy(q_values)
                nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
                state = nextstate
                episode_reward = episode_reward+reward
                cum_reward = cum_reward+reward
            reward_list.append(episode_reward)
        if stat:
            return cum_reward, reward_list
        else:
            return cum_reward
        
    def burn_in_memory_search(self,observationCombo,actionCombo,rewardCombo,
        observation_nextCombo,is_terminalCombo): # Burn-in with search
        print('Start burn in with search algorithm...')
        for i in range(len(observationCombo)):
            observation = observationCombo[i]
            action = actionCombo[i]
            reward = rewardCombo[i]
            observation_next = observation_nextCombo[i]
            is_terminal = is_terminalCombo[i]
            
        self.replay.store(observation, action, reward, observation_next, is_terminal, self.context.timestep)
        self.replay.is_burn_in = True
        print('Burn in with search algorithm finished.')

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')   # 定义接受的参数和它们的类型
	parser.add_argument('--env',dest='env',type=str)     # 环境名称
	parser.add_argument('--render',dest='render',type=int,default=0)    # 是否渲染环境，类型为整数，默认为0（不渲染）
	parser.add_argument('--train',dest='train',type=int,default=1)      # 是否进行训练，类型为整数，默认为1（进行训练）
	parser.add_argument('--test',dest='test',type=int,default=0)        # 是否进行测试，类型为整数，默认为0（不进行测试）
	parser.add_argument('--lookahead',dest='lookahead',type=int,default=0)     # 是否启用前瞻，类型为整数，默认为0（不启用）
	parser.add_argument('--test_final',dest='test_final',type=int,default=0)     # 是否进行最终测试，类型为整数，默认为0（不进行最终测试）
	parser.add_argument('--model_no',dest='model_file_no',type=str)     # 模型编号，类型为字符串
	return parser.parse_args()     # 解析这些参数，并返回解析后的参数对象

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DTQN_Agent class here, and then train / test it. 
	model_path = '../model/'
	data_path = '../data/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	agent = Agent(environment_name, sess, render=args.render)
	if args.train == 1:
		agent.train()
	if args.test == 1:
		print(agent.test(model_file="../model/model_{}.ckpt".format(args.model_file_no))/20.0)
	sess.close()


if __name__ == '__main__':
	main(sys.argv)


