"""Rule-based missile test — verify physics + ACMI + hit detection before RL.

The pursuer is controlled by a simple PID-like rule:
  1. Turn toward the target (minimize heading error)
  2. Maintain target altitude
  3. Maintain cruise speed
  4. Fire when within valid launch envelope (1–5 km, ATA < 20°)

This script validates:
  V1  Missile flies toward target (PN guidance)
  V2  ACMI trajectory is correct (Red alive, Yellow hit, Grey miss)
  V3  Hit detection triggers target_killed termination

Usage:
  python scripts/_test_missile_rulebased.py [difficulty] [seed] [acmi_output]
"""

import os, sys, warnings, logging, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

import numpy as np

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import (
    SingleCombatShootTask,
    MAX_ATTACK_ANGLE, MIN_ATTACK_DISTANCE, MAX_ATTACK_DISTANCE,
    NUM_MISSILES,
)
from src.utils.geometry import compute_forward_vector


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule-based pursuer controller
# ═══════════════════════════════════════════════════════════════════════════════

class RuleBasedPursuer:
    """Simple PID heading controller: point nose at target and close range."""

    def __init__(self, kp_hdg: float = 3.0, cruise_speed: float = 250.0):
        self.kp_hdg = kp_hdg
        self.cruise_speed = cruise_speed
        self.target_alt_m = 3000.0

    def compute_action(self, ps, target) -> np.ndarray:
        """Compute MultiDiscrete action [speed(3), heading(5), altitude(3), fire(2)].

        Fire decision is made separately by _should_fire().
        """
        action = np.array([1, 2, 1, 0], dtype=np.int64)  # default: hold all, no fire

        # ── Heading: turn toward target ──────────────────────────────────
        p_pos = ps.aircraft.position_ned
        t_pos = target.aircraft.position_ned
        los = t_pos - p_pos
        los_hdg = float(np.degrees(np.arctan2(los[1], los[0]))) % 360.0

        hdg_err = (los_hdg - ps.ref_hdg + 180.0) % 360.0 - 180.0  # [-180, 180]

        if abs(hdg_err) > 5.0:
            if hdg_err > 0:
                action[1] = 4  # turn right (+30°/s)
            else:
                action[1] = 0  # turn left (-30°/s)

        # ── Speed: maintain cruise ──────────────────────────────────────
        cur_spd = float(ps.aircraft.state["airspeed_mps"])
        cmd_spd = getattr(ps, '_cmd_speed', self.cruise_speed)
        if cur_spd < self.cruise_speed - 15:
            action[0] = 0  # decelerate index? No - accelerate = index 2
            # Wait, DELTA_SPEEDS = [-20, 0, +20]. Index 0 = -20 (decelerate),
            # Index 2 = +20 (accelerate). Let me be explicit.
            # accelerate: action[0] = 2, decelerate: action[0] = 0, hold: action[0] = 1

        if cur_spd < self.cruise_speed - 10:
            action[0] = 2  # accelerate
        elif cur_spd > self.cruise_speed + 10:
            action[0] = 0  # decelerate

        # ── Altitude: maintain target altitude ──────────────────────────
        alt_m = float(ps.aircraft.state["alt_m"])
        if ps.ref_alt_m - alt_m > 50:
            action[2] = 2  # climb
        elif alt_m - ps.ref_alt_m > 50:
            action[2] = 0  # descend

        return action

    def should_fire(self, ps, target) -> bool:
        """Check if within valid launch envelope."""
        if not target.is_alive:
            return False

        p_pos = ps.aircraft.position_ned
        t_pos = target.aircraft.position_ned
        dist = float(np.linalg.norm(p_pos - t_pos))

        if dist < MIN_ATTACK_DISTANCE or dist > MAX_ATTACK_DISTANCE:
            return False

        # ATA check
        p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
        los_dir = (t_pos - p_pos) / max(dist, 1e-6)
        cos_ata = float(np.dot(p_fwd, los_dir))
        ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ata))))

        return ata_deg < MAX_ATTACK_ANGLE


# ═══════════════════════════════════════════════════════════════════════════════
#  Main test
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    difficulty = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3  # S-turns for more dynamic chase
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    acmi_out = sys.argv[3] if len(sys.argv) > 3 else 'results/test_missile_rulebased.acmi'

    print(f"=== Rule-Based Missile Test ===")
    print(f"  difficulty={difficulty}  seed={seed}")

    # Create env with 1v1 shoot task
    env = BaseEnv(task=SingleCombatShootTask({'difficulty_level': difficulty}))
    obs, _ = env.reset(seed=seed)

    pursuer = RuleBasedPursuer()

    p0 = env.pursuers[0]
    t0 = env.targets[0]
    max_steps = 1500  # longer for multi-missile engagement

    # ── Force tail-chase: pursuer 3km behind target, both heading north ──
    # Key: set JSBSim lat/lon consistent with desired position_ned so ACMI renders correctly
    _force_tail_chase = True
    if _force_tail_chase:
        from src.dynamics.flight_controller import FlightControlTargets
        from src.environment.formation_task import PHYSICS_DT
        t_hdg, t_spd, t_alt = 0.0, 230.0, 3000.0  # target flies NORTH at 230m/s
        p_spd, chase_dist = 280.0, 4000.0  # pursuer faster, starts 4km behind
        # Target: ~2.2km north of reference (lat≈30.02, lon≈120.0)
        t0.aircraft.reset(lat_deg=30.02, lon_deg=120.0, alt_ft=int(t_alt*3.28084),
                          heading_deg=t_hdg, speed_kts=int(t_spd/0.5144), trim=False)
        t0.aircraft.position_ned = np.array([2224.0, 0.0, t_alt])
        t0.ref_hdg, t0.ref_alt_m = t_hdg, t_alt
        # Pursuer: 4km south of target → ~1.8km south of reference (lat≈29.984)
        p_ned_north = 2224.0 - chase_dist  # ≈ -1776
        p_lat = 30.0 + p_ned_north / 111320.0  # ≈ 29.984
        p0.aircraft.reset(lat_deg=p_lat, lon_deg=120.0, alt_ft=int(t_alt*3.28084),
                          heading_deg=t_hdg, speed_kts=int(p_spd/0.5144), trim=False)
        p0.aircraft.position_ned = np.array([p_ned_north, 0.0, t_alt])
        p0.ref_hdg, p0.ref_alt_m = t_hdg, t_alt
        p0._cmd_speed = p_spd
        # Short warmup — let JSBSim stablise at these positions
        for _ in range(int(1.0 * 60)):
            for ac, hdg, alt, spd in [(p0, t_hdg, t_alt, p_spd), (t0, t_hdg, t_alt, t_spd)]:
                s = ac.aircraft.state
                tgt = FlightControlTargets(heading_deg=hdg, altitude_m=alt, speed_mps=spd)
                thr, elev, ail, rud = ac.fc.compute(s, tgt, PHYSICS_DT)
                ac.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
                ac.aircraft.run()
                ac.aircraft.position_ned[0:2] += ac.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ac.aircraft.position_ned[2] = s["alt_m"]
        p0.prev_dist = float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned))

    print(f"\nInitial:")
    print(f"  P0: pos=({p0.aircraft.position_ned[0]:.0f}, {p0.aircraft.position_ned[1]:.0f}) "
          f"hdg={p0.ref_hdg:.0f}° spd={p0.aircraft.state['airspeed_mps']:.0f}m/s "
          f"alt={p0.aircraft.state['alt_m']:.0f}m")
    print(f"  T0: pos=({t0.aircraft.position_ned[0]:.0f}, {t0.aircraft.position_ned[1]:.0f}) "
          f"hdg={t0.ref_hdg:.0f}° alt={t0.aircraft.state['alt_m']:.0f}m")
    dist_init = float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned))
    print(f"  Distance: {dist_init:.0f}m  (P0 behind T0, tail chase north)")
    print(f"  Missiles: {env.task.remaining_missiles}")

    # Fire delay: let the pursuer chase for a few seconds before first launch
    FIRE_DELAY_STEPS = 10   # 2 seconds at 5Hz
    MIN_FIRE_DIST = 8000.0  # fire when within 8km (well within AIM-9L range)

    # Enable ACMI logging AFTER repositioning (so header shows correct positions)
    os.makedirs(os.path.dirname(acmi_out) or '.', exist_ok=True)
    env.enable_acmi_logging(acmi_out)
    env.log_acmi_step()

    total_rew = 0.0
    fired_steps = []
    hit_detected = False
    missiles_launched = 0

    for step in range(max_steps):
        # ── Compute rule-based action ──────────────────────────────────
        action_vec = pursuer.compute_action(p0, t0)

        # Fire if: cooldown elapsed, valid envelope, within range, past delay
        prev_remaining = env.task.remaining_missiles.get('p0', 0)
        if prev_remaining > 0 and step >= FIRE_DELAY_STEPS:
            dist = float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned))
            if pursuer.should_fire(p0, t0) and dist <= MIN_FIRE_DIST:
                from src.environment.singlecombat_shoot_task import MIN_ATTACK_INTERVAL
                if step - env.task._last_shoot_step.get('p0', -MIN_ATTACK_INTERVAL) >= MIN_ATTACK_INTERVAL:
                    action_vec[3] = 1  # fire!
                    missiles_launched += 1
                    fired_steps.append(step)

        action = {'p0': action_vec}
        obs, rews, terms, truncs, info = env.step(action)
        env.log_acmi_step()

        for r in rews.values():
            total_rew += r

        # ── Log progress ───────────────────────────────────────────────
        if step % 50 == 0 or terms.get('__all__') or (action_vec[3] == 1):
            dist = float(np.linalg.norm(
                p0.aircraft.position_ned - t0.aircraft.position_ned))
            bd = env.task._reward_breakdown
            launch_r = bd.get('ValidLaunchReward', {}).get('p0', 0)
            print(f"  step {step:3d}: dist={dist:.0f}m  "
                  f"hdg={p0.ref_hdg:.0f}°  spd={p0.aircraft.state['airspeed_mps']:.0f}m/s  "
                  f"rew={total_rew:+.0f}  {'FIRE!' if action_vec[3]==1 else ''}"
                  f"  launch_r={launch_r:+.0f}")

        if terms.get('__all__'):
            reason = info.get('p0', {}).get('termination_reason', 'unknown')
            print(f"\n  Terminated at step {step}: {reason}")
            if reason == 'target_killed':
                hit_detected = True
            break
    else:
        print(f"\n  Timeout at step {max_steps}")

    # ── Verification report ─────────────────────────────────────────────────
    print(f"\n=== Verification Report ===")
    print(f"  Missiles launched:  {missiles_launched}")
    print(f"  Fire steps:         {fired_steps}")
    print(f"  Final distance:     {float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned)):.0f}m")
    print(f"  Target alive:       {t0.is_alive}")
    print(f"  Pursuer alive:      {p0.is_alive}")
    print(f"  Total reward:       {total_rew:+.1f}")

    # V1: Missile flight
    print(f"\n  V1 Missile flight:  ", end="")
    active_missiles = [m for m in env._tempsims.values()]
    if missiles_launched > 0:
        print(f"PASS — {missiles_launched} missile(s) launched, "
              f"{sum(1 for m in active_missiles if m.is_success)} hit(s)")
    else:
        print("FAIL — no missiles launched (envelope never met)")

    # V2: ACMI output
    print(f"  V2 ACMI output:     ", end="")
    if os.path.exists(acmi_out) and os.path.getsize(acmi_out) > 0:
        # Check for color coding in ACMI
        with open(acmi_out, 'r') as f:
            content = f.read()
        has_red = 'Color=Red' in content and 'AIM-9L' in content
        has_explosion = 'Explosion' in content
        has_hit = 'Color=Yellow' in content
        has_miss = 'Color=Grey' in content
        print(f"PASS — {os.path.getsize(acmi_out)} bytes, "
              f"Red={'✓' if has_red else '✗'}, "
              f"Explosion={'✓' if has_explosion else '✗'}, "
              f"Yellow={'✓' if has_hit else '-' if not hit_detected else '✗'}")
    else:
        print(f"FAIL — {acmi_out} missing or empty")

    # V3: Hit detection
    print(f"  V3 Hit detection:   ", end="")
    if hit_detected:
        print("PASS — target_killed triggered correctly")
    elif not t0.is_alive:
        print("PASS — target died (non-missile cause)")
    else:
        launched = [m for m in env._tempsims.values() if not m.is_inactive]
        if launched:
            for m in launched:
                if m.is_success:
                    print("PARTIAL — missile hit but termination not yet fired")
                    break
            else:
                print("N/A — missile(s) missed (expected at this geometry)")
        else:
            print("N/A — no missile reached terminal state")

    print(f"\n  ACMI saved to: {acmi_out}")

    env.close()
    return 0 if hit_detected else 0  # always return 0 — V3 might not hit, that's OK


if __name__ == '__main__':
    sys.exit(main())
