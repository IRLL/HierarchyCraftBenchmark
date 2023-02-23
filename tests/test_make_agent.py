import pytest
import pytest_check as check

import gym

from craftbench.make_agent import load_agent, NAME_TO_AGENT


@pytest.mark.parametrize("agent_name", NAME_TO_AGENT.keys())
def test_each_agent(agent_name: str):
    env = gym.make("Acrobot-v1")
    net_arch = dict(pi=[8, 8], vf=[6, 6])
    agent = load_agent(
        agent_name=agent_name,
        env=env,
        policy_type="MlpPolicy",
        net_arch=net_arch,
        seed=0,
    )
    check.is_instance(agent, NAME_TO_AGENT[agent_name])
