"""低层神经飞控闭环跟踪测试 — 验证 NeuralFlightController 12维量纲适配。

不需要 RLlib，直接实例化 Aircraft + NeuralFlightController + SafetyInterceptor，
给定阶跃目标，观察飞机是否能平滑跟踪。

测试方案:
  0-5s:  保持直飞 (heading=当前, alt=当前, speed=当前)
  5-10s: 阶跃 (左转30°, 爬升100m, 加速到250m/s)
  10-15s: 反向阶跃 (右转30°回到原位, 下降100m, 减速到200m/s)
  15-20s: 保持

验证标准:
  - 副翼/升降舵/油门做出合理响应
  - 航向/高度/速度平滑逼近目标
  - 无剧烈抖动、倒飞或坠毁
"""

from __future__ import annotations

import os
import sys
import warnings
import logging
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)

from src.dynamics.aircraft import Aircraft
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.controller_base import FlightTarget
from src.dynamics.neural_controller import NeuralFlightController
from src.dynamics.safety_interceptor import SafetyInterceptor

CTRL_FREQ = 60.0
DT = 1.0 / CTRL_FREQ
DECISION_DT = 0.2  # 5 Hz
DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)  # 12


def run_tracking_test(controller_type: str = "neural", seed: int = 42):
    """Run step-response tracking test."""

    # ── Build aircraft ──────────────────────────────────────────────────
    ac = Aircraft()
    envelope = FlightEnvelope(EnvelopeConfig())

    rng = np.random.default_rng(seed)

    print(f"Controller: {controller_type}")

    # ── Build controller ────────────────────────────────────────────────
    if controller_type == "neural":
        controller = SafetyInterceptor(NeuralFlightController())
    elif controller_type == "pid":
        from src.dynamics.pid_controller import PIDFlightController
        controller = SafetyInterceptor(PIDFlightController())
    else:
        raise ValueError(f"Unknown controller_type: {controller_type}")

    # ── Reset aircraft ──────────────────────────────────────────────────
    init_hdg = rng.uniform(0, 360)
    init_alt_ft = 20000
    init_speed_kts = 400

    ac.reset(
        lat_deg=30.0, lon_deg=120.0,
        alt_ft=init_alt_ft, heading_deg=init_hdg,
        speed_kts=init_speed_kts, trim=False)
    ac.position_ned = np.array([0.0, 0.0, -3000.0])
    controller.reset(initial_speed_mps=init_speed_kts * 0.514)

    # Warmup: 3s level flight to settle trim
    print("  Warming up (3s level flight)...")
    for _ in range(int(3.0 * CTRL_FREQ)):
        tgt = FlightTarget(heading_deg=init_hdg, altitude_m=3000.0,
                           speed_mps=init_speed_kts * 0.514)
        surfaces = controller.predict(ac.state, tgt, DT)
        ac.set_controls(throttle=surfaces.throttle,
                        elevator=surfaces.elevator,
                        aileron=surfaces.aileron,
                        rudder=surfaces.rudder)
        ac.run()
        ac.position_ned[0:2] += ac.velocity_ned[0:2] * DT
        ac.position_ned[2] = ac.state["alt_m"]

    current_hdg = float(ac.state["yaw_deg"])
    current_alt = float(ac.state["alt_m"])
    current_spd = float(ac.state["airspeed_mps"])

    print(f"  Post-warmup: hdg={current_hdg:.1f}° alt={current_alt:.0f}m spd={current_spd:.1f}m/s")

    # ── Define step command sequence ─────────────────────────────────────
    target_sequence = [
        # (start_time_s, end_time_s, hdg_change, alt_change, spd_target)
        (0.0,  5.0,    0.0,    0.0, current_spd),        # hold
        (5.0,  10.0, -30.0, +100.0, 250.0),               # left turn + climb + accelerate
        (10.0, 15.0, +30.0, -100.0, 200.0),               # right turn back + descend + decel
        (15.0, 20.0,   0.0,    0.0, 220.0),               # settle
    ]

    # ── Run test ────────────────────────────────────────────────────────
    total_time = 20.0
    total_steps = int(total_time * CTRL_FREQ)
    decision_interval = DECISION_STEPS

    log = {
        "time": [], "hdg": [], "alt": [], "spd": [],
        "thr": [], "elev": [], "ail": [], "rud": [],
        "target_hdg": [], "target_alt": [], "target_spd": [],
    }

    base_hdg = current_hdg
    base_alt = current_alt
    base_spd = current_spd

    tgt_hdg = base_hdg
    tgt_alt = base_alt
    tgt_spd = base_spd

    print("\n  Running step-response test...")
    for step in range(total_steps):
        t = step * DT

        # Determine target from sequence
        for t_start, t_end, dh, da, ds in target_sequence:
            if t_start <= t < t_end:
                tgt_hdg = (base_hdg + dh) % 360.0
                tgt_alt = base_alt + da
                tgt_spd = ds
                break

        # Apply control at decision rate
        if step % decision_interval == 0:
            target = FlightTarget(heading_deg=tgt_hdg, altitude_m=tgt_alt,
                                  speed_mps=tgt_spd)
            surfaces = controller.predict(ac.state, target, DT)

        ac.set_controls(throttle=surfaces.throttle,
                        elevator=surfaces.elevator,
                        aileron=surfaces.aileron,
                        rudder=surfaces.rudder)
        ac.run()
        ac.position_ned[0:2] += ac.velocity_ned[0:2] * DT
        ac.position_ned[2] = ac.state["alt_m"]

        # Check crash
        if float(ac.state["alt_m"]) < 10.0:
            print(f"  ❌ CRASH at t={t:.1f}s — altitude below 10m!")
            break

        # Log
        log["time"].append(t)
        log["hdg"].append(float(ac.state["yaw_deg"]))
        log["alt"].append(float(ac.state["alt_m"]))
        log["spd"].append(float(ac.state["airspeed_mps"]))
        log["thr"].append(surfaces.throttle)
        log["elev"].append(surfaces.elevator)
        log["ail"].append(surfaces.aileron)
        log["rud"].append(surfaces.rudder)
        log["target_hdg"].append(tgt_hdg)
        log["target_alt"].append(tgt_alt)
        log["target_spd"].append(tgt_spd)

    try:
        ac.close()
    except AttributeError:
        pass

    # ── Analyze ──────────────────────────────────────────────────────────
    times = np.array(log["time"])
    hdg_arr = np.array(log["hdg"])
    alt_arr = np.array(log["alt"])
    spd_arr = np.array(log["spd"])
    tgt_hdg_arr = np.array(log["target_hdg"])
    tgt_alt_arr = np.array(log["target_alt"])
    tgt_spd_arr = np.array(log["target_spd"])

    # Heading error (circular, unwrapped)
    hdg_err = np.abs((hdg_arr - tgt_hdg_arr + 180) % 360 - 180)

    print(f"\n  Final state: hdg={hdg_arr[-1]:.1f}° alt={alt_arr[-1]:.0f}m spd={spd_arr[-1]:.1f}m/s")
    print(f"  Steady-state errors (last 1s):")
    mask = times > 19.0
    print(f"    heading MAE: {hdg_err[mask].mean():.2f}°")
    print(f"    altitude MAE: {np.abs(alt_arr[mask] - tgt_alt_arr[mask]).mean():.1f}m")
    print(f"    speed MAE:    {np.abs(spd_arr[mask] - tgt_spd_arr[mask]).mean():.1f}m/s")

    # Check stability — no oscillations
    hdg_smooth = np.std(np.diff(hdg_arr)) < 2.0
    alt_smooth = np.std(np.diff(alt_arr)) < 5.0
    spd_smooth = np.std(np.diff(spd_arr)) < 10.0

    if hdg_smooth and alt_smooth and spd_smooth:
        print("\n  ✅ PASS: Smooth tracking, no oscillations, no crash")
    else:
        print(f"\n  ⚠️  Stability check: hdg_smooth={hdg_smooth} alt_smooth={alt_smooth} spd_smooth={spd_smooth}")

    # ── Export CSV for plotting ─────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "neural_tracking")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"tracking_{controller_type}.csv")
    with open(csv_path, "w") as f:
        f.write("time,hdg,target_hdg,hdg_err,alt,target_alt,spd,target_spd,thr,elev,ail,rud\n")
        for i in range(len(times)):
            f.write(f"{times[i]:.2f},{hdg_arr[i]:.2f},{tgt_hdg_arr[i]:.2f},{hdg_err[i]:.2f},"
                    f"{alt_arr[i]:.1f},{tgt_alt_arr[i]:.1f},{spd_arr[i]:.1f},{tgt_spd_arr[i]:.1f},"
                    f"{log['thr'][i]:.4f},{log['elev'][i]:.4f},{log['ail'][i]:.4f},{log['rud'][i]:.4f}\n")
    print(f"  CSV saved to: {csv_path}")

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="neural",
                        choices=["neural", "pid"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("NeuralFlightController Closed-Loop Tracking Test")
    print("=" * 60)
    run_tracking_test(controller_type=args.controller, seed=args.seed)
