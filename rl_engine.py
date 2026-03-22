"""
Medallion-X Reinforcement Learning Engine
PPO/DQN hybrid implementation for trading decisions
Production-ready RL with continuous learning and adaptation
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque, namedtuple
import random
import time
import json

from ray.rllib.agents import ppo
from ray.rllib.agents import dqn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray import tune
import gymnasium as gym
from gymnasium import spaces

import redis.asyncio as redis

from ..config.settings import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Named tuple for experience replay
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class TradingState:
    """Trading state representation for RL"""
    # Market features
    price_features: np.ndarray
    volume_features: np.ndarray
    technical_features: np.ndarray
    
    # Sentiment features
    news_sentiment: float
    social_sentiment: float
    
    # On-chain features
    onchain_metrics: np.ndarray
    
    # Portfolio state
    current_position: float  # Current position size
    unrealized_pnl: float
    portfolio_value: float
    
    # Market state
    volatility: float
    trend_strength: float
    market_regime: int  # 0: ranging, 1: trending, 2: volatile
    
    # Risk metrics
    var_95: float  # Value at Risk
    max_drawdown: float
    sharpe_ratio: float

@dataclass
class TradingAction:
    """Trading action representation"""
    action_type: int  # 0: hold, 1: buy, 2: sell
    position_size: float  # Position size (0-1)
    leverage: float  # Leverage multiplier
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class RLConfig:
    """RL configuration parameters"""
    # PPO parameters
    ppo_lr: float = 3e-4
    ppo_clip_ratio: float = 0.2
    ppo_value_loss_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_epochs: int = 10
    ppo_batch_size: int = 64
    
    # DQN parameters
    dqn_lr: float = 1e-4
    dqn_gamma: float = 0.99
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.01
    dqn_epsilon_decay: float = 0.995
    dqn_target_update_freq: int = 1000
    dqn_memory_size: int = 100000
    
    # Hybrid parameters
    hybrid_weight_ppo: float = 0.6
    hybrid_weight_dqn: float = 0.4
    
    # Training parameters
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    save_freq: int = 100
    eval_freq: int = 50

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL training
    - Simulates market dynamics
    - Provides realistic reward structure
    - Handles position management
    """
    
    def __init__(self, config: RLConfig):
        super().__init__()
        
        self.config = config
        self.current_step = 0
        self.episode_step = 0
        
        # Action space: [action_type, position_size, leverage]
        # action_type: 0=hold, 1=buy, 2=sell
        # position_size: 0-1 (normalized)
        # leverage: 1-10 (normalized to 0-1)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([2, 1, 1]),
            dtype=np.float32
        )
        
        # State space (will be defined dynamically based on features)
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(100,),  # Will be adjusted based on actual features
            dtype=np.float32
        )
        
        # Trading state
        self.reset_state()
        
        # Market data simulation
        self.price_history = deque(maxlen=1000)
        self.current_price = 50000.0
        self.volatility = 0.02
        
    def reset_state(self) -> None:
        """Reset trading state"""
        self.portfolio_value = 10000.0
        self.current_position = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.max_portfolio_value = 10000.0
        self.trade_history = []
        self.episode_step = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        self.reset_state()
        self.current_step = 0
        
        # Generate initial market conditions
        self.current_price = 50000.0 + random.uniform(-1000, 1000)
        self.volatility = random.uniform(0.01, 0.05)
        
        state = self._get_state()
        info = {'episode': 0, 'step': 0}
        
        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.episode_step += 1
        self.current_step += 1
        
        # Parse action
        action_type = int(action[0])
        position_size = action[1]
        leverage = 1.0 + action[2] * 9.0  # 1-10x leverage
        
        # Execute action
        reward = self._execute_action(action_type, position_size, leverage)
        
        # Update market
        self._update_market()
        
        # Get new state
        state = self._get_state()
        
        # Check termination
        done = self._check_done()
        truncated = False
        
        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'current_position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'current_price': self.current_price
        }
        
        return state, reward, done, truncated, info

    def _execute_action(self, action_type: int, position_size: float, leverage: float) -> float:
        """Execute trading action and calculate reward"""
        old_portfolio_value = self.portfolio_value
        old_position = self.current_position
        
        # Execute action based on type
        if action_type == 1:  # Buy
            new_position = self.current_position + (position_size * leverage)
        elif action_type == 2:  # Sell
            new_position = self.current_position - (position_size * leverage)
        else:  # Hold
            new_position = self.current_position
        
        # Calculate transaction costs
        transaction_cost = abs(new_position - old_position) * self.current_price * 0.001  # 0.1% cost
        
        # Update position
        self.current_position = max(-1.0, min(1.0, new_position))  # Limit position size
        
        # Calculate PnL
        price_change = (self.current_price - 50000.0) / 50000.0  # Relative to initial price
        self.unrealized_pnl = self.current_position * price_change * self.portfolio_value
        
        # Update portfolio value
        self.portfolio_value = 10000.0 + self.unrealized_pnl - transaction_cost
        
        # Calculate reward
        reward = self._calculate_reward(old_portfolio_value, transaction_cost)
        
        # Record trade
        if action_type in [1, 2]:
            self.trade_history.append({
                'step': self.episode_step,
                'action_type': action_type,
                'position_size': position_size,
                'leverage': leverage,
                'price': self.current_price,
                'portfolio_value': self.portfolio_value
            })
        
        return reward

    def _update_market(self) -> None:
        """Update market conditions"""
        # Simulate price movement (random walk with volatility)
        price_change = np.random.normal(0, self.volatility)
        self.current_price *= (1 + price_change)
        
        # Update volatility (mean reversion)
        self.volatility = 0.9 * self.volatility + 0.1 * random.uniform(0.01, 0.05)
        
        # Store price history
        self.price_history.append(self.current_price)

    def _calculate_reward(self, old_portfolio_value: float, transaction_cost: float) -> float:
        """Calculate reward for the action"""
        # Portfolio return
        portfolio_return = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Risk-adjusted return
        risk_penalty = abs(self.current_position) * self.volatility * 0.1
        
        # Transaction cost penalty
        cost_penalty = transaction_cost / old_portfolio_value
        
        # Drawdown penalty
        if self.portfolio_value < self.max_portfolio_value:
            drawdown_penalty = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value * 0.5
        else:
            self.max_portfolio_value = self.portfolio_value
            drawdown_penalty = 0.0
        
        # Total reward
        reward = portfolio_return - risk_penalty - cost_penalty - drawdown_penalty
        
        return reward

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state_features = []
        
        # Price features (normalized)
        if len(self.price_history) > 0:
            prices = np.array(list(self.price_history))
            returns = np.diff(prices) / prices[:-1]
            
            # Recent returns
            recent_returns = returns[-10:] if len(returns) >= 10 else np.zeros(10)
            state_features.extend(recent_returns)
            
            # Price position
            if len(prices) >= 20:
                price_min, price_max = np.min(prices[-20:]), np.max(prices[-20:])
                price_position = (self.current_price - price_min) / (price_max - price_min) if price_max > price_min else 0.5
            else:
                price_position = 0.5
            state_features.append(price_position)
        else:
            state_features.extend([0.0] * 11)  # 10 returns + position
        
        # Technical indicators (simplified)
        if len(self.price_history) >= 14:
            # RSI
            price_changes = np.diff(prices)
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rsi = 100 - (100 / (1 + avg_gain/avg_loss)) if avg_loss > 0 else 50
            state_features.append(rsi / 100)  # Normalized
        else:
            state_features.append(0.5)
        
        # Portfolio state
        state_features.append(self.current_position)  # Current position
        state_features.append(self.unrealized_pnl / self.portfolio_value)  # Normalized PnL
        state_features.append(self.portfolio_value / 10000.0 - 1)  # Normalized portfolio value
        
        # Market state
        state_features.append(self.volatility)  # Current volatility
        
        # Time features
        state_features.append(self.episode_step / 1000.0)  # Progress in episode
        
        # Pad to fixed size
        while len(state_features) < 100:
            state_features.append(0.0)
        
        return np.array(state_features[:100], dtype=np.float32)

    def _check_done(self) -> bool:
        """Check if episode should terminate"""
        # Portfolio loss threshold
        if self.portfolio_value < 5000.0:  # 50% loss
            return True
        
        # Maximum steps
        if self.episode_step >= self.config.max_steps_per_episode:
            return True
        
        # Position limit breach
        if abs(self.current_position) > 1.5:
            return True
        
        return False

class HybridPolicyNetwork(nn.Module):
    """
    Hybrid neural network combining PPO and DQN architectures
    - Shared feature extraction layers
    - Separate policy and value heads (PPO)
    - Q-value head (DQN)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extraction
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # PPO policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # PPO value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # DQN Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through network"""
        # Extract features
        features = self.feature_layers(state)
        
        # PPO outputs
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        # DQN outputs
        q_values = self.q_head(features)
        
        return {
            'policy_logits': policy_logits,
            'value': value,
            'q_values': q_values
        }

class PPOAgent:
    """PPO agent implementation"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.policy_net = HybridPolicyNetwork(state_dim, action_dim)
        self.value_net = HybridPolicyNetwork(state_dim, action_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.ppo_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.ppo_lr)
        
        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Metrics
        self.metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'kl_divergence': 0.0,
            'updates': 0
        }

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action using PPO policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.policy_net(state_tensor)
            policy_logits = outputs['policy_logits']
            value = outputs['value']
        
        # Sample action from policy
        action_probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.numpy(), log_prob.item(), value.item()

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float, 
                         value: float, log_prob: float, done: bool) -> None:
        """Store experience for PPO update"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def update(self) -> Dict[str, float]:
        """Update PPO agent"""
        if len(self.states) < self.config.ppo_batch_size:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        values = torch.FloatTensor(np.array(self.values))
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        dones = torch.BoolTensor(np.array(self.dones))
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, values, dones)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(self.config.ppo_epochs):
            # Sample minibatch
            indices = torch.randperm(len(states))[:self.config.ppo_batch_size]
            
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_returns = returns[indices]
            batch_advantages = advantages[indices]
            batch_old_log_probs = log_probs[indices]
            
            # Policy update
            outputs = self.policy_net(batch_states)
            policy_logits = outputs['policy_logits']
            
            action_probs = F.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # PPO ratio
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # PPO loss
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.config.ppo_entropy_coef * entropy
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Value update
            value_pred = outputs['value'].squeeze()
            value_loss = F.mse_loss(value_pred, batch_returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Clear experience
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        # Update metrics
        self.metrics['policy_loss'] = total_policy_loss / self.config.ppo_epochs
        self.metrics['value_loss'] = total_value_loss / self.config.ppo_epochs
        self.metrics['entropy'] = total_entropy / self.config.ppo_epochs
        self.metrics['updates'] += 1
        
        return self.metrics

    def _compute_returns(self, rewards: torch.Tensor, values: torch.Tensor, 
                        dones: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.config.dqn_gamma * running_return
            returns[t] = running_return
        
        return returns

class DQNAgent:
    """DQN agent implementation"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.q_net = HybridPolicyNetwork(state_dim, action_dim)
        self.target_net = HybridPolicyNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.dqn_lr)
        
        # Experience replay
        self.memory = deque(maxlen=config.dqn_memory_size)
        
        # Epsilon
        self.epsilon = config.dqn_epsilon_start
        
        # Metrics
        self.metrics = {
            'q_loss': 0.0,
            'epsilon': self.epsilon,
            'updates': 0
        }

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select action using DQN epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random action
            action = np.random.randint(0, self.action_dim)
            q_value = 0.0
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                outputs = self.q_net(state_tensor)
                q_values = outputs['q_values']
            action = q_values.argmax().item()
            q_value = q_values.max().item()
        
        return np.array([action]), q_value

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))

    def update(self) -> Dict[str, float]:
        """Update DQN agent"""
        if len(self.memory) < self.config.ppo_batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.memory, self.config.ppo_batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor(np.array([e.action for e in batch]))
        rewards = torch.FloatTensor(np.array([e.reward for e in batch]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.BoolTensor(np.array([e.done for e in batch]))
        
        # Current Q-values
        current_q_values = self.q_net(states)['q_values'].gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)['q_values'].max(1)[0]
            target_q_values = rewards + (self.config.dqn_gamma * next_q_values * ~dones)
        
        # Compute loss
        q_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.config.dqn_epsilon_end, self.epsilon * self.config.dqn_epsilon_decay)
        
        # Update target network
        if self.metrics['updates'] % self.config.dqn_target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Update metrics
        self.metrics['q_loss'] = q_loss.item()
        self.metrics['epsilon'] = self.epsilon
        self.metrics['updates'] += 1
        
        return self.metrics

class HybridRLAgent:
    """
    Hybrid RL agent combining PPO and DQN
    - Ensemble decision making
    - Adaptive weighting
    - Continuous learning
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Individual agents
        self.ppo_agent = PPOAgent(state_dim, action_dim, config)
        self.dqn_agent = DQNAgent(state_dim, action_dim, config)
        
        # Performance tracking
        self.ppo_performance = deque(maxlen=100)
        self.dqn_performance = deque(maxlen=100)
        
        # Adaptive weights
        self.ppo_weight = config.hybrid_weight_ppo
        self.dqn_weight = config.hybrid_weight_dqn
        
        # Metrics
        self.metrics = {
            'total_updates': 0,
            'ppo_weight': self.ppo_weight,
            'dqn_weight': self.dqn_weight,
            'hybrid_performance': 0.0
        }

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Select action using hybrid policy"""
        # Get actions from both agents
        ppo_action, ppo_log_prob, ppo_value = self.ppo_agent.select_action(state)
        dqn_action, dqn_q_value = self.dqn_agent.select_action(state)
        
        # Combine actions (weighted average)
        if random.random() < self.ppo_weight:
            final_action = ppo_action
            confidence = abs(ppo_value)
        else:
            final_action = dqn_action
            confidence = abs(dqn_q_value)
        
        action_info = {
            'ppo_action': ppo_action[0],
            'dqn_action': dqn_action[0],
            'final_action': final_action[0],
            'confidence': confidence,
            'ppo_weight': self.ppo_weight,
            'dqn_weight': self.dqn_weight
        }
        
        return final_action, action_info

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool, **kwargs) -> None:
        """Store experience for both agents"""
        # Store for PPO
        self.ppo_agent.store_experience(
            state, action, reward, 
            kwargs.get('value', 0.0),
            kwargs.get('log_prob', 0.0),
            done
        )
        
        # Store for DQN
        self.dqn_agent.store_experience(state, action, reward, next_state, done)

    def update(self) -> Dict[str, Any]:
        """Update both agents and adapt weights"""
        # Update individual agents
        ppo_metrics = self.ppo_agent.update()
        dqn_metrics = self.dqn_agent.update()
        
        # Adaptive weight adjustment based on recent performance
        self._adapt_weights()
        
        # Combined metrics
        combined_metrics = {
            'ppo': ppo_metrics,
            'dqn': dqn_metrics,
            'hybrid': self.metrics
        }
        
        self.metrics['total_updates'] += 1
        
        return combined_metrics

    def _adapt_weights(self) -> None:
        """Adapt hybrid weights based on performance"""
        # Simple adaptation based on recent loss values
        if len(self.ppo_agent.metrics) > 0 and len(self.dqn_agent.metrics) > 0:
            ppo_loss = self.ppo_agent.metrics.get('policy_loss', 1.0)
            dqn_loss = self.dqn_agent.metrics.get('q_loss', 1.0)
            
            # Lower loss = better performance
            total_loss = ppo_loss + dqn_loss
            if total_loss > 0:
                new_ppo_weight = dqn_loss / total_loss
                new_dqn_weight = ppo_loss / total_loss
                
                # Smooth adaptation
                self.ppo_weight = 0.9 * self.ppo_weight + 0.1 * new_ppo_weight
                self.dqn_weight = 0.9 * self.dqn_weight + 0.1 * new_dqn_weight
                
                # Normalize
                total_weight = self.ppo_weight + self.dqn_weight
                self.ppo_weight /= total_weight
                self.dqn_weight /= total_weight
                
                self.metrics['ppo_weight'] = self.ppo_weight
                self.metrics['dqn_weight'] = self.dqn_weight

class RLEngine:
    """
    Main RL engine for Medallion-X
    - Coordinates training and inference
    - Manages multiple agents
    - Handles continuous learning
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.config = RLConfig()
        
        # Environment
        self.env = TradingEnvironment(self.config)
        
        # Agents for different symbols
        self.agents: Dict[str, HybridRLAgent] = {}
        
        # Training state
        self.is_training = False
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            'episodes_completed': 0,
            'total_steps': 0,
            'average_reward': 0.0,
            'best_reward': -float('inf'),
            'training_time': 0.0
        }

    def get_or_create_agent(self, symbol: str) -> HybridRLAgent:
        """Get or create agent for a symbol"""
        if symbol not in self.agents:
            state_dim = self.env.state_space.shape[0]
            action_dim = 3  # hold, buy, sell
            
            self.agents[symbol] = HybridRLAgent(state_dim, action_dim, self.config)
            logger.info(f"Created RL agent for {symbol}")
        
        return self.agents[symbol]

    async def train_agent(self, symbol: str, episodes: int = None) -> Dict[str, Any]:
        """Train agent for a specific symbol"""
        if episodes is None:
            episodes = self.config.max_episodes
        
        agent = self.get_or_create_agent(symbol)
        self.is_training = True
        
        training_rewards = []
        start_time = time.time()
        
        for episode in range(episodes):
            if not self.is_training:
                break
            
            episode_reward = 0.0
            state, _ = self.env.reset()
            
            for step in range(self.config.max_steps_per_episode):
                # Select action
                action, action_info = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Store experience
                agent.store_experience(
                    state, action, reward, next_state, done,
                    value=action_info.get('confidence', 0.0),
                    log_prob=0.0  # Simplified
                )
                
                episode_reward += reward
                state = next_state
                
                if done or truncated:
                    break
            
            # Update agent
            agent_metrics = agent.update()
            
            # Record metrics
            training_rewards.append(episode_reward)
            self.metrics['episodes_completed'] += 1
            self.metrics['total_steps'] += step + 1
            
            # Update best reward
            if episode_reward > self.metrics['best_reward']:
                self.metrics['best_reward'] = episode_reward
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(training_rewards[-10:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best: {self.metrics['best_reward']:.2f}")
                
                # Store in Redis
                await self._store_training_metrics(symbol, episode, avg_reward, agent_metrics)
        
        # Calculate final metrics
        self.metrics['average_reward'] = np.mean(training_rewards)
        self.metrics['training_time'] = time.time() - start_time
        
        self.is_training = False
        
        return {
            'symbol': symbol,
            'episodes': episodes,
            'average_reward': self.metrics['average_reward'],
            'best_reward': self.metrics['best_reward'],
            'training_time': self.metrics['training_time'],
            'agent_metrics': agent.get_metrics()
        }

    async def predict_action(self, symbol: str, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get action prediction for a state"""
        agent = self.get_or_create_agent(symbol)
        action, action_info = agent.select_action(state)
        
        return action, action_info

    async def _store_training_metrics(self, symbol: str, episode: int, reward: float, 
                                    agent_metrics: Dict[str, Any]) -> None:
        """Store training metrics in Redis"""
        metrics_data = {
            'symbol': symbol,
            'episode': episode,
            'reward': reward,
            'agent_metrics': agent_metrics,
            'timestamp': int(time.time() * 1000)
        }
        
        key = f"rl:training:{symbol}:latest"
        await self.redis_client.setex(
            key,
            ttl=86400,  # 24 hours
            value=json.dumps(metrics_data, default=str)
        )
        
        # Store in time series
        ts_key = f"rl:training_ts:{symbol}"
        await self.redis_client.zadd(
            ts_key,
            {json.dumps(metrics_data, default=str): episode}
        )
        # Keep only last 1000 episodes
        await self.redis_client.zremrangebyrank(ts_key, 0, -1001)

    async def get_training_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training history for a symbol"""
        try:
            ts_key = f"rl:training_ts:{symbol}"
            data_points = await self.redis_client.zrevrange(ts_key, 0, limit - 1)
            
            history = []
            for data in data_points:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                history.append(json.loads(data))
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving training history: {e}")
            return []

    async def start(self) -> None:
        """Start RL engine"""
        self.is_running = True
        logger.info("RL engine started")

    async def stop(self) -> None:
        """Stop RL engine"""
        self.is_running = False
        self.is_training = False
        logger.info("RL engine stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        base_metrics = self.metrics.copy()
        
        # Add agent-specific metrics
        agent_metrics = {}
        for symbol, agent in self.agents.items():
            agent_metrics[symbol] = agent.get_metrics()
        
        base_metrics['agents'] = agent_metrics
        base_metrics['total_agents'] = len(self.agents)
        
        return base_metrics
