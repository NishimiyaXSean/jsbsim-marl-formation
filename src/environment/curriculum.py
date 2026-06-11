"""Curriculum learning scheduler for progressive difficulty."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CurriculumConfig:
    """Per-stage difficulty parameters."""

    stage: int
    evader_speed_coeff: float   # fraction of attacker's max speed
    evader_g_coeff: float        # fraction of attacker's max G
    warning_radius: float        # evader becomes reactive when attacker is within this (m)


CURRICULUM_STAGES = {
    1: CurriculumConfig(stage=1, evader_speed_coeff=0.65, evader_g_coeff=0.333, warning_radius=1500.0),
    2: CurriculumConfig(stage=2, evader_speed_coeff=0.75, evader_g_coeff=0.55, warning_radius=3000.0),
    3: CurriculumConfig(stage=3, evader_speed_coeff=0.85, evader_g_coeff=0.85, warning_radius=10000.0),
}

# Eval: attacker must achieve this success rate on a fixed-pattern target to advance
TARGET_SUCCESS_RATE = 0.70
EVAL_INTERVAL = 10       # evaluate every N training iterations
TEST_EPISODES = 50        # number of eval episodes per check


def get_curriculum(stage: int) -> CurriculumConfig:
    """Return curriculum config for the given stage (1-indexed)."""
    return CURRICULUM_STAGES.get(stage, CURRICULUM_STAGES[3])


def should_advance(stage: int, eval_success_rate: float, max_stage: int = 3) -> bool:
    """Check if the agent is ready to advance to the next stage."""
    return eval_success_rate >= TARGET_SUCCESS_RATE and stage < max_stage
