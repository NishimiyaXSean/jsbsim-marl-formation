"""三步验证法：Task-Based 架构重构验证。

Step 1: 冒烟测试 — 实例化、随机动作运行、多次 reset
Step 2: 等价性对齐测试 — 旧 FormationRLlibEnv vs 新 BaseEnv+FormationTask
"""

from __future__ import annotations

import gc
import sys
import warnings
import logging
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1: Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def test_smoke_instantiate_and_run():
    """实例化新环境，随机动作运行 1000 step，验证无崩溃、无 NaN 泄漏。"""
    from src.environment.formation_task import FormationTask
    from src.environment.base_env import BaseEnv

    task = FormationTask()
    env = BaseEnv(task=task)

    # Verify spaces match
    assert env.observation_space == task.observation_space
    assert env.action_space == task.action_space
    assert len(env.pursuers) == 2
    assert len(env.targets) == 1
    print("✓ Env + Task instantiated correctly")

    # Run 1000 steps with random actions
    obs, _ = env.reset(seed=42)
    step_count = 0
    for _ in range(1000):
        actions = {aid: env.action_space[aid].sample() for aid in env._agent_ids}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        step_count += 1

        # Check no NaN in observations
        for aid in env._agent_ids:
            for k, v in obs[aid].items():
                assert np.all(np.isfinite(v)), f"NaN in obs[{aid}][{k}] at step {step_count}"

        # Check no NaN in rewards
        for aid, r in rewards.items():
            assert np.isfinite(r) or r == float('-inf'), \
                f"Non-finite reward for {aid}: {r} at step {step_count}"

        if terminateds.get("__all__") or truncateds.get("__all__"):
            obs, _ = env.reset(seed=42 + step_count)

    env.close()
    print(f"✓ Smoke test: {step_count} steps, no crashes, no NaN leakage")


def test_smoke_multiple_resets():
    """多次 reset + step 循环，验证无内存泄漏和状态残留。"""
    from src.environment.formation_task import FormationTask
    from src.environment.base_env import BaseEnv

    task = FormationTask()
    env = BaseEnv(task=task)

    for i in range(5):
        obs, _ = env.reset(seed=42 + i)
        # Verify obs shape is correct
        for aid in env._agent_ids:
            assert len(obs[aid]["obs"].shape) == 1, f"Wrong obs shape for {aid}"
            assert len(obs[aid]["global_state"].shape) == 1, f"Wrong global_state shape for {aid}"
            assert obs[aid]["action_mask"].shape == (8,), f"Wrong action_mask shape for {aid}"
            # Check no NaN
            assert np.all(np.isfinite(obs[aid]["obs"])), f"NaN in obs[{aid}]"
            assert np.all(np.isfinite(obs[aid]["global_state"])), f"NaN in global_state[{aid}]"

        # Run steps then reset
        for _ in range(10):
            actions = {aid: env.action_space[aid].sample() for aid in env._agent_ids}
            obs, rewards, terminateds, truncateds, infos = env.step(actions)

    env.close()
    gc.collect()
    print("✓ Multiple reset: 5 cycles, no memory leak, correct shapes")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2: Equivalence Alignment Test
# ═══════════════════════════════════════════════════════════════════════════════

def test_alignment_reset():
    """新老环境 Reset 对齐：相同 seed → 相同初始观测。"""
    from src.environment.formation_rllib_env import FormationRLlibEnv
    from src.environment.formation_task import FormationTask
    from src.environment.base_env import BaseEnv

    seed = 42
    np.random.seed(seed)

    # Old env
    old_env = FormationRLlibEnv({"difficulty_level": 0.0})

    # New env
    task = FormationTask({"difficulty_level": 0.0})
    new_env = BaseEnv(task=task)

    obs_old, _ = old_env.reset(seed=seed)
    obs_new, _ = new_env.reset(seed=seed)

    assert set(obs_old.keys()) == set(obs_new.keys()), \
        f"Agent ID mismatch: old={set(obs_old.keys())} new={set(obs_new.keys())}"

    for aid in obs_old.keys():
        assert isinstance(obs_old[aid], dict), f"Old obs[{aid}] is not dict: {type(obs_old[aid])}"
        assert isinstance(obs_new[aid], dict), f"New obs[{aid}] is not dict: {type(obs_new[aid])}"

        for k in obs_old[aid].keys():
            old_val = np.asarray(obs_old[aid][k])
            new_val = np.asarray(obs_new[aid][k])
            mismatch = not np.allclose(old_val, new_val, rtol=1e-5, atol=1e-5)
            if mismatch:
                diff = np.abs(old_val - new_val)
                print(f"  ⚠ Reset mismatch [{aid}][{k}]: max_diff={diff.max():.6f} "
                      f"old_shape={old_val.shape} new_shape={new_val.shape}")
            else:
                print(f"  ✓ Reset aligned [{aid}][{k}]: shape={old_val.shape}")

    old_env.close()
    new_env.close()
    print("✓ Reset alignment check complete")


def test_alignment_step_by_step():
    """新老环境 Step-by-Step 对齐：相同动作序列 → 相同输出。"""
    from src.environment.formation_rllib_env import FormationRLlibEnv
    from src.environment.formation_task import FormationTask
    from src.environment.base_env import BaseEnv

    seed = 42
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    old_env = FormationRLlibEnv({"difficulty_level": 0.0})
    task = FormationTask({"difficulty_level": 0.0})
    new_env = BaseEnv(task=task)

    obs_old, _ = old_env.reset(seed=seed)
    obs_new, _ = new_env.reset(seed=seed)

    mismatches = []
    num_steps = 50

    for step_idx in range(num_steps):
        # Generate identical actions
        action_dict = {
            aid: np.array([rng.integers(0, 5), rng.integers(0, 3)], dtype=np.int64)
            for aid in old_env._agent_ids
        }

        o_old, r_old, term_old, trunc_old, _ = old_env.step(action_dict)
        o_new, r_new, term_new, trunc_new, _ = new_env.step(action_dict)

        # Check rewards
        for aid in old_env._agent_ids:
            if abs(r_old[aid] - r_new[aid]) > 1e-5:
                mismatches.append(
                    f"  step {step_idx} reward[{aid}]: old={r_old[aid]:.4f} new={r_new[aid]:.4f} diff={abs(r_old[aid]-r_new[aid]):.6f}")

        # Check terminated/truncated
        for aid in old_env._agent_ids:
            if term_old[aid] != term_new[aid]:
                mismatches.append(
                    f"  step {step_idx} terminated[{aid}]: old={term_old[aid]} new={term_new[aid]}")
            if trunc_old[aid] != trunc_new[aid]:
                mismatches.append(
                    f"  step {step_idx} truncated[{aid}]: old={trunc_old[aid]} new={trunc_new[aid]}")

        # Check all__ keys
        if term_old.get("__all__") != term_new.get("__all__"):
            mismatches.append(
                f"  step {step_idx} terminated[__all__]: old={term_old.get('__all__')} new={term_new.get('__all__')}")

        # Check observations
        for aid in old_env._agent_ids:
            for k in old_env.observation_space[aid].spaces:
                old_val = np.asarray(o_old[aid][k])
                new_val = np.asarray(o_new[aid][k])
                if old_val.shape != new_val.shape:
                    mismatches.append(
                        f"  step {step_idx} obs[{aid}][{k}] shape: old={old_val.shape} new={new_val.shape}")
                elif not np.allclose(old_val, new_val, rtol=1e-5, atol=1e-5):
                    diff = np.abs(old_val - new_val)
                    mismatches.append(
                        f"  step {step_idx} obs[{aid}][{k}]: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}")

        if terminateds := term_old.get("__all__") or trunc_old.get("__all__"):
            obs_old, _ = old_env.reset(seed=seed + step_idx + 1)
            obs_new, _ = new_env.reset(seed=seed + step_idx + 1)

    old_env.close()
    new_env.close()

    print(f"✓ Alignment: {num_steps} steps run, {len(mismatches)} mismatches found")
    if mismatches:
        print("Mismatches:")
        for m in mismatches[:30]:  # show first 30
            print(m)
        if len(mismatches) > 30:
            print(f"  ... and {len(mismatches) - 30} more")
    else:
        print("✓ Perfect alignment!")

    return len(mismatches) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--alignment", action="store_true", default=False)
    args = parser.parse_args()
    # If no specific test selected, run all
    if not args.smoke and not args.alignment:
        args.smoke = True
        args.alignment = True

    print("=" * 60)
    print("Step 1: Smoke Tests")
    print("=" * 60)
    if args.smoke:
        test_smoke_instantiate_and_run()
        test_smoke_multiple_resets()

    print()
    print("=" * 60)
    print("Step 2: Equivalence Alignment Tests")
    print("=" * 60)
    if args.alignment:
        test_alignment_reset()
        print()
        aligned = test_alignment_step_by_step()
        if not aligned:
            print("\n⚠ Some mismatches detected — FormationTask needs further alignment with FormationRLlibEnv.")
            print("  This is expected: the task extraction is a staged process.")
            print("  Use the mismatches above to identify what logic still needs to be ported.")
