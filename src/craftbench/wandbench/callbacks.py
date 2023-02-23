from typing import Optional

import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback


class Sb3WandbCallback(WandbCallback):
    def __init__(
        self,
        max_n_consecutive_successes: Optional[int] = None,
        infos_log_freq: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.infos_log_freq = infos_log_freq
        self.n_successes = 0
        self.n_consecutive_successes = 0
        self.max_n_consecutive_successes = max_n_consecutive_successes
        self._purpose_is_done = False

    def _should_log_infos(self):
        if self.infos_log_freq and self.model.num_timesteps % self.infos_log_freq == 0:
            return True
        return False

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
        infos: dict = self.locals.get("infos")[0]
        purpose_is_done = infos.get("Purpose is done", False)
        metrics = {}
        if self._should_log_infos():
            metrics.update(infos)
        if purpose_is_done:
            self._purpose_is_done = True
        if env_done:
            if self._purpose_is_done:
                self.n_successes += 1
                self.n_consecutive_successes += 1
            else:
                self.n_consecutive_successes = 0
            self._purpose_is_done = False
            metrics.update(
                {
                    "n_successes": self.n_successes,
                    "n_consecutive_successes": self.n_consecutive_successes,
                },
            )
        if metrics:
            wandb.log(metrics, step=self.model.num_timesteps)
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
