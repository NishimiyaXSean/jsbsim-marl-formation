"""Open-loop / rule-based controller test for the 1v1 shoot task.

Separates controller quality from RL learning:
- The pursuer is commanded by a RULE: heading = bearing to target,
  altitude = target altitude, speed = fixed command — the exact
  FlightTarget interface the RL agent drives through
  SafetyInterceptor(PIDFlightController).
- Target flies straight (difficulty=0.0, same reset geometry as training).
- Reports: min/end distance, tracking errors, bank, closing rate,
  termination reason.

Usage:
  python scripts/test_openloop_controller.py --episodes 5 --speeds 220,250,280 --seed 42
"""
import os, sys, warnings, logging, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

import numpy as np

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.environment.formation_task import PHYSICS_DT, DECISION_STEPS
from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.dynamics.controller_base import FlightTarget
from src.dynamics.pid_controller import PIDFlightController
from src.dynamics.safety_interceptor import SafetyInterceptor
from src.utils.units import kts_to_mps

MAX_STEPS = 1500
CATCH_DIST = 300.0


def bearing_deg(p_pos, t_pos):
    """Absolute heading (0=N, CW) from p_pos to t_pos."""
    dx = t_pos[1] - p_pos[1]
    dy = t_pos[0] - p_pos[0]
    return float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)


def chase_rule(p0, t0, cmd_speed, turn_speed=200.0):
    """Lead-pursuit rule with speed management.

    Aims at the target's predicted position (lead_time based on range) and
    slows down when a large heading correction is needed (slower = faster
    turn rate at fixed bank).
    """
    p_pos = p0.aircraft.position_ned
    t_pos = t0.aircraft.position_ned
    t_vel = t0.aircraft.velocity_ned
    dist = float(np.linalg.norm(p_pos - t_pos))
    # Lead shrinks smoothly to pure pursuit near the target — a fixed 1s lead
    # creates a stable side-by-side orbit at ~700m (closing ~11 m/s only).
    lead_time = float(np.clip(dist / 1000.0 - 0.5, 0.0, 5.0))
    aim = t_pos + t_vel * lead_time
    hdg = bearing_deg(p_pos, aim)
    s = p0.aircraft.state
    hdg_err = abs((hdg - s["yaw_deg"] + 180.0) % 360.0 - 180.0)
    speed = cmd_speed if hdg_err < 15.0 else max(turn_speed, cmd_speed - 60.0)
    return hdg, speed


def set_geometry(env, p0, t0, chase_dist, heading_bias, t_spd, p_spd):
    """Deterministic tail-chase geometry + 1s warmup (reproducible sweeps)."""
    t_hdg, t_alt = 0.0, 3000.0
    t0.aircraft.reset(lat_deg=30.02, lon_deg=120.0, alt_ft=int(t_alt * 3.28084),
                      heading_deg=t_hdg, speed_kts=int(t_spd / 0.5144), trim=False)
    t0.aircraft.position_ned = np.array([2224.0, 0.0, t_alt])
    t0.ref_hdg, t0.ref_alt_m = t_hdg, t_alt

    p_hdg = (t_hdg + heading_bias) % 360.0
    p_ned_north = 2224.0 - chase_dist
    p_lat = 30.0 + p_ned_north / 111320.0
    p0.aircraft.reset(lat_deg=p_lat, lon_deg=120.0, alt_ft=int(t_alt * 3.28084),
                      heading_deg=p_hdg, speed_kts=int(p_spd / 0.5144), trim=False)
    p0.aircraft.position_ned = np.array([p_ned_north, 0.0, t_alt])
    p0.ref_hdg, p0.ref_alt_m = p_hdg, t_alt
    p0._cmd_speed = p_spd

    for _ in range(60):  # 1s warmup through the RL-used controller chain
        s = p0.aircraft.state
        target = FlightTarget(heading_deg=p_hdg, altitude_m=t_alt, speed_mps=p_spd)
        surfaces = p0.controller.predict(s, target, PHYSICS_DT)
        p0.aircraft.set_controls(
            throttle=float(np.clip(surfaces.throttle, 0.0, 1.0)),
            elevator=float(np.clip(surfaces.elevator, -1.0, 1.0)),
            aileron=float(np.clip(surfaces.aileron, -1.0, 1.0)),
            rudder=float(np.clip(surfaces.rudder, -1.0, 1.0)))
        ts = t0.aircraft.state
        tgt = FlightControlTargets(heading_deg=t_hdg, altitude_m=t_alt,
                                   speed_mps=t_spd)
        thr, elev, ail, rud = t0.fc.compute(ts, tgt, PHYSICS_DT)
        t0.aircraft.set_controls(thr, elev, ail, rud)
        p0.aircraft.run()
        p0.aircraft.position_ned[0:2] += p0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        p0.aircraft.position_ned[2] = p0.aircraft.state["alt_m"]
        t0.aircraft.run()
        t0.aircraft.position_ned[0:2] += t0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        t0.aircraft.position_ned[2] = t0.aircraft.state["alt_m"]


def run_episode(seed, cmd_speed=250.0, max_steps=MAX_STEPS, fbw=False,
                bank_ff=0.30, max_bank=55.0, kd_q=0.8, roll_per_deg=2.5,
                turn_speed=200.0, chase_dist=4000.0, heading_bias=30.0,
                trace=False, alt_kd=0.002, alt_kp=0.008):
    env = BaseEnv(task=SingleCombatShootTask({"difficulty_level": 0.0}))
    obs, _ = env.reset(seed=seed)
    p0, t0 = env.pursuers[0], env.targets[0]
    # Swap in a controller with the sweep parameters (RL path untouched).
    p0.controller = SafetyInterceptor(PIDFlightController(
        bank_ff_gain=bank_ff, kd_q=kd_q,
        max_bank_deg=max_bank, roll_per_deg_heading=roll_per_deg,
        alt_kd=alt_kd, alt_kp=alt_kp))
    if fbw:
        # Bypass the F-16 FCS (its yaw damper blocks turns); drive surfaces directly.
        p0.aircraft.fdm["fcs/fbw-override"] = 1
    set_geometry(env, p0, t0, chase_dist, heading_bias, 200.0, cmd_speed)

    dt = PHYSICS_DT
    min_dist = float("inf")
    dists = []
    hdg_errs, alt_errs, spd_errs, banks = [], [], [], []
    step = 0
    reason = "timeout"

    for step in range(max_steps):
        p_pos = p0.aircraft.position_ned
        t_pos = t0.aircraft.position_ned
        dist = float(np.linalg.norm(p_pos - t_pos))
        dists.append(dist)
        min_dist = min(min_dist, dist)
        if dist < CATCH_DIST:
            reason = "caught"
            break

        target_alt = float(t0.aircraft.state["alt_m"])
        hdg_cmd, spd_cmd = chase_rule(p0, t0, cmd_speed, turn_speed=turn_speed)
        target = FlightTarget(
            heading_deg=hdg_cmd,
            altitude_m=target_alt,
            speed_mps=spd_cmd,
        )

        # 12 substeps — identical loop to BaseEnv.step
        for _ in range(DECISION_STEPS):
            s = p0.aircraft.state
            surfaces = p0.controller.predict(s, target, dt)
            p0.aircraft.set_controls(
                throttle=float(np.clip(surfaces.throttle, 0.0, 1.0)),
                elevator=float(np.clip(surfaces.elevator, -1.0, 1.0)),
                aileron=float(np.clip(surfaces.aileron, -1.0, 1.0)),
                rudder=float(np.clip(surfaces.rudder, -1.0, 1.0)))
            ts = t0.aircraft.state
            tgt = FlightControlTargets(heading_deg=t0.ref_hdg, altitude_m=3000.0,
                                       speed_mps=kts_to_mps(310))
            thr, elev, ail, rud = t0.fc.compute(ts, tgt, dt)
            t0.aircraft.set_controls(thr, elev, ail, rud)

            p0.aircraft.run()
            p0.aircraft.position_ned[0:2] += p0.aircraft.velocity_ned[0:2] * dt
            p0.aircraft.position_ned[2] = p0.aircraft.state["alt_m"]
            t0.aircraft.run()
            t0.aircraft.position_ned[0:2] += t0.aircraft.velocity_ned[0:2] * dt
            t0.aircraft.position_ned[2] = t0.aircraft.state["alt_m"]

        s = p0.aircraft.state
        hdg_errs.append((bearing_deg(p_pos, t_pos) - s["yaw_deg"] + 180.0) % 360.0 - 180.0)
        alt_errs.append(abs(s["alt_m"] - target_alt))
        spd_errs.append(s["airspeed_mps"] - cmd_speed)
        banks.append(abs(s["roll_deg"]))
        if trace and step % 25 == 0:
            print(f"    TRACE t={step*0.2:5.1f}s dist={dist:6.0f}m "
                  f"alt_err={abs(s['alt_m']-target_alt):5.0f}m "
                  f"hdg_err={hdg_errs[-1]:+6.1f}° bank={s['roll_deg']:+5.1f}° "
                  f"p_spd={s['airspeed_mps']:5.0f} "
                  f"t_spd={t0.aircraft.state['airspeed_mps']:5.0f} "
                  f"pitch={s['pitch_deg']:+5.1f}° nz={s['n_z_g']:+5.2f}g "
                  f"elev_cmd={surfaces.elevator:+5.2f}")

        if s["alt_m"] < 1000.0:
            reason = "low_altitude"
            break
        if dist > 15000.0:
            reason = "lost_target"
            break
        if abs(s["alt_m"] - t0.aircraft.state["alt_m"]) > 3000.0:
            reason = "fled_combat_altitude"
            break

    env.close()
    n = max(step + 1, 1)
    return {
        "seed": seed, "steps": step + 1, "reason": reason,
        "start_dist": dists[0],
        "min_dist": min_dist,
        "end_dist": dists[-1],
        "mean_abs_hdg_err_deg": float(np.mean(np.abs(hdg_errs))) if hdg_errs else 0.0,
        "mean_alt_err_m": float(np.mean(alt_errs)) if alt_errs else 0.0,
        "mean_spd_err_mps": float(np.mean(spd_errs)) if spd_errs else 0.0,
        "mean_bank_deg": float(np.mean(banks)) if banks else 0.0,
        "closing_rate_mps": (dists[0] - dists[-1]) / n * 5.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Open-loop controller tracking test")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--speeds", type=str, default="250",
                        help="comma-separated commanded speeds, e.g. 220,250,280")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fbw", action="store_true",
                        help="set fcs/fbw-override=1 on the pursuer (bypass FCS yaw damper)")
    parser.add_argument("--bank-ff", type=float, default=0.25,
                        help="bank-turn elevator feedforward gain (pull-up magnitude)")
    parser.add_argument("--max-bank", type=float, default=75.0,
                        help="maximum bank angle for the heading loop (deg)")
    parser.add_argument("--kd-q", type=float, default=2.0,
                        help="pitch-rate damping gain on elevator")
    parser.add_argument("--roll-per-deg", type=float, default=2.5,
                        help="deg bank per deg heading error")
    parser.add_argument("--turn-speed", type=float, default=200.0,
                        help="speed used while correcting a large heading error")
    parser.add_argument("--chase-dist", type=float, default=4000.0,
                        help="initial tail-chase distance (m)")
    parser.add_argument("--alt-kd", type=float, default=0.04,
                        help="altitude PID derivative gain (vertical-speed damping)")
    parser.add_argument("--alt-kp", type=float, default=0.008,
                        help="altitude PID proportional gain")
    parser.add_argument("--heading-bias", type=float, default=30.0,
                        help="initial pursuer heading offset from target (deg)")
    parser.add_argument("--trace", action="store_true",
                        help="print per-step metrics for the first episode")
    args = parser.parse_args()

    speeds = [float(x) for x in args.speeds.split(",") if x.strip()]
    tag = "fbw" if args.fbw else "fcs"
    for spd in speeds:
        rows = [run_episode(args.seed + i, cmd_speed=spd, fbw=args.fbw,
                            bank_ff=args.bank_ff, max_bank=args.max_bank,
                            kd_q=args.kd_q, roll_per_deg=args.roll_per_deg,
                            turn_speed=args.turn_speed,
                            chase_dist=args.chase_dist,
                            heading_bias=args.heading_bias,
                            trace=args.trace and i == 0,
                            alt_kd=args.alt_kd, alt_kp=args.alt_kp)
                for i in range(args.episodes)]
        print(f"\n===== [{tag}] speed={spd:.0f} bank_ff={args.bank_ff:.2f} "
              f"max_bank={args.max_bank:.0f} kd_q={args.kd_q:.2f} "
              f"roll_per_deg={args.roll_per_deg:.1f} turn_speed={args.turn_speed:.0f} "
              f"({args.episodes} eps) =====")
        for r in rows:
            print(f"  ep{rows.index(r):>2d} seed={r['seed']:<4d} {r['reason']:<8s} "
                  f"steps={r['steps']:>4d} start={r['start_dist']:>6.0f}m "
                  f"min={r['min_dist']:>6.0f}m end={r['end_dist']:>6.0f}m "
                  f"hdg={r['mean_abs_hdg_err_deg']:>5.1f}° alt={r['mean_alt_err_m']:>5.0f}m "
                  f"spd={r['mean_spd_err_mps']:>+5.0f} bank={r['mean_bank_deg']:>4.1f}° "
                  f"closing={r['closing_rate_mps']:>+5.0f}m/s")
        caught = sum(1 for r in rows if r["reason"] == "caught")
        mean_min = float(np.mean([r["min_dist"] for r in rows]))
        mean_hdg = float(np.mean([r["mean_abs_hdg_err_deg"] for r in rows]))
        mean_alt = float(np.mean([r["mean_alt_err_m"] for r in rows]))
        mean_spd = float(np.mean([r["mean_spd_err_mps"] for r in rows]))
        print(f"  -> caught {caught}/{args.episodes}  mean_min={mean_min:.0f}m  "
              f"mean|hdg_err|={mean_hdg:.1f}°  mean_alt_err={mean_alt:.0f}m  "
              f"mean_spd_err={mean_spd:+.0f}m/s")


if __name__ == "__main__":
    main()