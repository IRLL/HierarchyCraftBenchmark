import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from hcraft import HcraftEnv


def record_wrap_env(hcraft_env: HcraftEnv, video_path: str):
    env = Monitor(hcraft_env)  # record stats such as returns
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        video_path,
        record_video_trigger=lambda step: (step >= 1000)
        and (int(np.log10(step + 1)) - int(np.log10(step))),
        video_length=hcraft_env.max_step,
    )
    return env
