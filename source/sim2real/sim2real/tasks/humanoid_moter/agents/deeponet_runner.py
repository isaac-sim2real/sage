import torch
from typing import Dict, List, Tuple
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from .deeponet_actor_critic import DeepONetActorCritic
import wandb
import numpy as np

class DeepONetRunner(OnPolicyRunner):
    """Runner for training DeepONet-based policies"""
    
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        self.train_cfg = train_cfg
        # Override the actor critic with DeepONet version
        self.actor_critic = DeepONetActorCritic(
            branch_input_dims=train_cfg["policy"]["branch_input_dims"],
            branch_hidden_dim=train_cfg["policy"]["branch_hidden_dim"],
            trunk_input_dim=train_cfg["policy"]["trunk_input_dim"],
            trunk_hidden_dims=train_cfg["policy"]["trunk_hidden_dims"],
            action_dim=train_cfg["policy"]["action_dim"],
            critic_hidden_dims=train_cfg["policy"]["critic_hidden_dims"],
            activation=train_cfg["policy"]["activation"]
        ).to(self.device)
        
        # Initialize wandb if logger is set to wandb
        if self.train_cfg["logger"] == "wandb":
            wandb.init(
                project=self.train_cfg["wandb_project"],
                config=self.train_cfg,
                name=self.train_cfg["experiment_name"]
            )
        
    def _compute_returns(self, last_values, dones, rewards, values, actions, log_probs, advantages):
        """Override to handle DeepONet-specific return computation"""
        # Split the observations into branch and trunk inputs
        branch_inputs = [obs[:, :dim] for obs, dim in zip(self.obs, self.train_cfg["policy"]["branch_input_dims"])]
        trunk_input = self.obs[-1][:, -self.train_cfg["policy"]["trunk_input_dim"]:]
        
        # Compute returns using DeepONet architecture
        with torch.no_grad():
            last_values = self.actor_critic.critic(trunk_input)
            
        returns = self._compute_gae_returns(
            last_values, dones, rewards, values, 
            self.train_cfg["algorithm"]["gamma"], 
            self.train_cfg["algorithm"]["lam"]
        )
        
        return returns, advantages
    
    def _update_network(self, obs_dict, returns, advantages, actions, log_probs):
        """Override to handle DeepONet-specific network updates"""
        # Split observations for DeepONet
        branch_inputs = [obs[:, :dim] for obs, dim in zip(obs_dict['obs'], self.train_cfg["policy"]["branch_input_dims"])]
        trunk_input = obs_dict['obs'][-1][:, -self.train_cfg["policy"]["trunk_input_dim"]:]
        
        # Initialize metrics for logging
        metrics = {
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'mean_returns': returns.mean().item(),
            'mean_advantages': advantages.mean().item(),
            'mean_values': self.actor_critic.critic(trunk_input).mean().item()
        }
        
        # Update network using DeepONet architecture
        for _ in range(self.train_cfg["algorithm"]["num_learning_epochs"]):
            for indices in self._get_mini_batch_indices():
                # Get mini-batch data
                mb_branch_inputs = [x[indices] for x in branch_inputs]
                mb_trunk_input = trunk_input[indices]
                mb_returns = returns[indices]
                mb_advantages = advantages[indices]
                mb_actions = actions[indices]
                mb_old_log_probs = log_probs[indices]
                
                # Forward pass
                values, new_log_probs, entropy = self.actor_critic.evaluate(
                    mb_branch_inputs, mb_trunk_input, mb_actions
                )
                
                # Compute losses
                value_loss = self._compute_value_loss(values, mb_returns)
                policy_loss = self._compute_policy_loss(new_log_probs, mb_old_log_probs, mb_advantages)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss * self.train_cfg["algorithm"]["policy_loss_coef"] +
                    value_loss * self.train_cfg["algorithm"]["value_loss_coef"] +
                    entropy_loss * self.train_cfg["algorithm"]["entropy_coef"]
                )
                
                # Update metrics
                metrics['value_loss'] += value_loss.item()
                metrics['policy_loss'] += policy_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()
                metrics['total_loss'] += loss.item()
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                if self.train_cfg["algorithm"]["max_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), 
                        self.train_cfg["algorithm"]["max_grad_norm"]
                    )
                self.optimizer.step()
                
                # Update learning rate if needed
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
        
        # Average metrics over all updates
        num_updates = self.train_cfg["algorithm"]["num_learning_epochs"] * self.train_cfg["algorithm"]["num_mini_batches"]
        for key in ['value_loss', 'policy_loss', 'entropy_loss', 'total_loss']:
            metrics[key] /= num_updates
        
        # Log metrics to wandb if enabled
        if self.train_cfg["logger"] == "wandb":
            wandb.log(metrics, step=self.iter)
        
        # Print metrics in the same format as original task
        print(f"\nIteration {self.iter}:")
        print(f"Value Loss: {metrics['value_loss']:.4f}")
        print(f"Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"Entropy Loss: {metrics['entropy_loss']:.4f}")
        print(f"Total Loss: {metrics['total_loss']:.4f}")
        print(f"Mean Returns: {metrics['mean_returns']:.4f}")
        print(f"Mean Advantages: {metrics['mean_advantages']:.4f}")
        print(f"Mean Values: {metrics['mean_values']:.4f}\n")
    
    def _get_mini_batch_indices(self):
        """Get indices for mini-batch training"""
        batch_size = self.train_cfg["algorithm"]["num_steps_per_env"] * self.env.num_envs
        mini_batch_size = batch_size // self.train_cfg["algorithm"]["num_mini_batches"]
        indices = torch.randperm(batch_size)
        for i in range(0, batch_size, mini_batch_size):
            yield indices[i:i + mini_batch_size]