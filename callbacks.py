from typing import Optional

import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback


class WandbCallback(WandbCallback):
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        max_n_consecutive_successes: Optional[int] = None,
    ):
        super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq)
        self.n_successes = 0
        self.n_consecutive_successes = 0
        self.max_n_consecutive_successes = max_n_consecutive_successes
        self._tasks_done = False

    def _on_step(self):
        # Checking for both 'done' and 'dones' keywords because:
        # Some models use keyword 'done' (e.g.,: SAC, TD3, DQN, DDPG)
        # While some models use keyword 'dones' (e.g.,: A2C, PPO)
        done = np.array(
            self.locals.get("done")
            if self.locals.get("done") is not None
            else self.locals.get("dones")
        )
        env_done = np.any(done)
        tasks_done = self.locals.get("infos")[0].get("tasks_done")
        if tasks_done:
            self._tasks_done = True
        if env_done:
            if self._tasks_done:
                self.n_successes += 1
                self.n_consecutive_successes += 1
            else:
                self.n_consecutive_successes = 0
            self._tasks_done = False
            wandb.log(
                {
                    "n_successes": self.n_successes,
                    "n_consecutive_successes": self.n_consecutive_successes,
                },
                step=self.model.num_timesteps,
            )
        if (
            self.max_n_consecutive_successes is not None
            and self.n_consecutive_successes >= self.max_n_consecutive_successes
        ):
            return False
        return True

    def _on_rollout_end(self):
        mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        mean_lenght = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        current_step = self.model.num_timesteps
        wandb.log(
            {"mean_ep_return": mean_reward, "mean_ep_lenght": mean_lenght},
            step=current_step,
        )
