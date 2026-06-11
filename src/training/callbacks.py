"""RLlib callbacks for tracking air combat metrics."""

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class AirCombatCallbacks(DefaultCallbacks):
    """Track per-episode kill/crash/OOB/timeout rates."""

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for("attacker_0")
        reason = info.get("reason", "timeout") if info else "timeout"

        episode.hist_data["rate_success"] = [1.0 if reason == "success" else 0.0]
        episode.hist_data["rate_crash"] = [1.0 if reason == "ground_crash" else 0.0]
        episode.hist_data["rate_oob"] = [1.0 if reason == "out_of_bounds" else 0.0]
        episode.hist_data["rate_timeout"] = [1.0 if reason == "timeout" else 0.0]
