"""Vendored OpenAI Baselines (stable-baselines v2.10.0), trimmed to what BALANCE uses.

Only PPO2 (see ``stable_baselines/ppo2/ppo2_BALANCE.py``) and the shared
``common``/``bench`` infrastructure are kept; the other algorithm
implementations of the upstream library were removed.
"""
from stable_baselines import logger  # noqa: F401
from stable_baselines.ppo2 import PPO2  # noqa: F401

__version__ = "2.10.0"
