"""Recompute every headline number straight from the live artifacts.

The paper's numbers have been corrected several times; this script is the
cheap way to prove the draft and the artifacts still agree. It reads only the
JSON artifacts and each file's own run_meta -- never a filename-derived
assumption about which model produced what.

Run:  python scripts/audit_paper_numbers.py
      python scripts/audit_paper_numbers.py path/to/extracted_pdf_text.txt

Passing a path is how the post-typesetting check works (2026-09-19): the build
script runs pdftotext over the compiled PDF and audits THAT text, because a
number can survive the source and still be mangled by typesetting -- 46.50%
becoming 46.50 \%, an exponent slipping into math font, a table cell wrapping.

The E3-CLR artifact is optional on purpose: while the fresh-env expert-CLR run
is still going, the audit stays useful, and the moment the artifact appears two
extra obligations switch on -- the measured value must be quoted in the draft,
and the {{EXPERT_CLR}} placeholder must be gone.

Note on scope: the "in text" checks confirm a value is PRESENT somewhere in the
audited text. They cannot detect a value that is present but attached to the
wrong claim, so they are a floor, not a substitute for reading the paper.
"""
from __future__ import annotations

import io
import json
import re
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir))
EV = os.path.join(ROOT, "results", "shoot_eval")
PAPER = os.path.join(ROOT, "paper", "small_paper_draft.md")


def load(name):
    with io.open(os.path.join(EV, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []

    def add(tag, computed, artifact):
        rows.append((tag, computed, artifact))

    def pct(d, key="kill_rate", raw=False, label=None):
        """Format a rate defensively: schemas differ between scripts."""
        if raw:
            ci, ca = d.get("clr_fire_commands"), d.get("clr_allowed_steps")
            if d.get("clr") is None:
                return "ABSENT"
            return "%.2f%% (%s/%s)" % (100 * d["clr"], ci, ca)
        if d.get(key) is None:
            return "ABSENT"
        n = d.get("episodes") or d.get("n") or d.get("n_episodes")
        if n:
            return "%.2f%% (%d/%d)" % (100 * d[key], round(d[key] * n), n)
        return "%.2f%%" % (100 * d[key])

    # --- E7: BC vs SPC at d=0 ------------------------------------------------
    e7b, e7s = load("E7_bc_round1_d0_n400_s20000.json"), load("E7_spc_d0_n400_s20000.json")
    add("E7 BC kill", pct(e7b), "E7_bc_round1_d0_n400_s20000.json")
    add("E7 SPC kill", pct(e7s), "E7_spc_d0_n400_s20000.json")
    add("E7 BC CLR", pct(e7b, raw=True), "same")
    add("E7 SPC CLR", pct(e7s, raw=True), "same")

    # --- E7 paired (derived file; its own schema, not the two-arm one) -------
    try:
        p7 = load("E7_paired_bc_vs_spc_d0_n400.json")
        add("E7 paired delta", "%+.2f pp" % p7["kill_rate"]["paired_diff_pp"],
            "E7_paired_bc_vs_spc_d0_n400.json")
        c = p7["contingency"]
        add("E7 discordant (favouring B : A)",
            "%d : %d" % (c["discordant_favouring_b"], c["only_a_kill"]), "same")
        add("E7 exact McNemar p", "%.4g" % p7["mcnemar"]["exact_two_sided_p"], "same")
        add("E7 chi2-cc p", "%.4g" % p7["mcnemar"]["chi2_continuity_corrected_p"], "same")
    except Exception as exc:                                  # noqa: BLE001
        add("E7 paired", "UNREADABLE: %s" % exc, "-")

    # --- E5: d=0.3 -----------------------------------------------------------
    e5b, e5s = load("E5_bc_round1_d03_n400_s20000.json"), load("E5_spc_d03_n400_s20000.json")
    add("E5 BC kill @0.3", pct(e5b), "E5_bc_round1_d03_n400_s20000.json")
    add("E5 SPC kill @0.3", pct(e5s), "E5_spc_d03_n400_s20000.json")
    add("E5 BC CLR @0.3", pct(e5b, raw=True), "same")
    add("E5 SPC CLR @0.3", pct(e5s, raw=True), "same")

    # --- E3: fresh-env expert vs BC -----------------------------------------
    for tag, fn in (("d=0", "E3_paired_bc_vs_expert_d0_n400_s20000_v2.json"),
                    ("d=0.3", "E3_paired_bc_vs_expert_d03_n400_s20000.json")):
        d = load(fn)
        add("E3 %s expert kill" % tag, "%.2f%%" % (100 * d["expert"]["kill_rate"]), fn)
        add("E3 %s BC kill" % tag, "%.2f%%" % (100 * d["bc"]["kill_rate"]), fn)
        add("E3 %s paired delta" % tag, "%+.2f pp" % (100 * d["paired"]["delta_kill_mean"]), fn)
        add("E3 %s W/L/T" % tag, str(d["paired"]["delta_kill_win_lose_tie"]), fn)
        add("E3 %s env_lifecycle" % tag,
            str(d.get("run_meta", {}).get("env_lifecycle", "ABSENT")), fn)
        add("E3 %s expert CLR" % tag,
            ("%.2f%%" % (100 * d["expert"]["clr"])
             if "clr" in d.get("expert", {})
             else "ABSENT in this file (see the E3-CLR block below)"), fn)

    # --- E1 / E6: frozen-trajectory oracle ----------------------------------
    def oracle_rate(doc, name, art):
        """E1/E6 store one entry per oracle; each holds an 'aggregate' block."""
        node = doc.get(name)
        if not isinstance(node, dict):
            add("oracle %s" % name, "ABSENT", art)
            return
        agg = node.get("aggregate", node)
        kr = agg.get("kill_rate")
        if kr is None:
            add("oracle %s" % name, "ABSENT", art)
            return
        n = agg.get("episodes") or agg.get("n")
        add("oracle %s" % name,
            ("%.1f%% (%d/%d)" % (100 * kr, round(kr * n), n)) if n else "%.1f%%" % (100 * kr),
            art)

    e1 = load("E1_fire_oracle_dist2_3k_s60.json")
    for k in ("asap", "delay_30", "delay_60", "dlz_mid", "dlz_deep",
              "interval_100", "bc"):
        oracle_rate(e1, k, "E1_fire_oracle_dist2_3k_s60.json")

    e6 = load("E6_fire_oracle_target_evasive_s60.json")
    for k in ("asap", "bc"):
        oracle_rate(e6, k, "E6_fire_oracle_target_evasive_s60.json")

    # --- E2: rule oracle ----------------------------------------------------
    try:
        e2 = load("E2_asap_oracle_id_n400_s20000.json")
        node = e2.get("id_400", e2)
        if isinstance(node, dict) and node.get("kill_rate") is not None:
            add("E2 rule oracle kill", pct(node), "E2_asap_oracle_id_n400_s20000.json")
        else:
            add("E2 rule oracle kill", "ABSENT", "E2_asap_oracle_id_n400_s20000.json")
    except Exception as exc:                                  # noqa: BLE001
        add("E2 rule oracle", "UNREADABLE: %s" % exc, "-")

    # --- E3-CLR: the fresh-env expert CLR (the paper's last open gap) --------
    # Deliberately optional: the run takes ~1.7 h, so the audit must stay
    # useful while it is in flight. When the artifact lands, two extra
    # obligations switch on automatically: the measured value must appear in
    # the draft, and the {{EXPERT_CLR}} placeholder must be gone.
    expert_clr_pct = None
    clr_values = {}  # tag -> expert CLR in percent, only for completed runs
    clr_artifacts = (
        ("d=0", "E3_paired_bc_vs_expert_d0_n400_s20000_v3_clr.json"),
        ("d=0.3", "E3_paired_bc_vs_expert_d03_n400_s20000_v2_clr.json"),
    )
    for tag, clr_name in clr_artifacts:
        clr_path = os.path.join(EV, clr_name)
        if not os.path.exists(clr_path):
            add("E3-CLR %s expert CLR" % tag,
                "ABSENT (run in flight or not started)", clr_name)
            continue
        d = load(clr_name)
        done = bool(d.get("run_meta", {}).get("complete", False))
        value = 100.0 * float(d["expert"]["clr"])
        add("E3-CLR %s expert CLR" % tag,
            "%.2f%% (%d/%d)" % (value, d["expert"]["clr_fire_commands"],
                                d["expert"]["clr_allowed_steps"]), clr_name)
        add("E3-CLR %s BC CLR" % tag,
            "%.2f%% (%d/%d)" % (100.0 * d["bc"]["clr"], d["bc"]["clr_fire_commands"],
                                d["bc"]["clr_allowed_steps"]), clr_name)
        add("E3-CLR %s run state" % tag,
            "complete" if done else "IN FLIGHT (%s/%s episodes)"
            % (d.get("run_meta", {}).get("episodes_completed", "?"),
               d.get("run_meta", {}).get("n_episodes", "?")), clr_name)
        if not done:
            print("note: %s exists but is not complete; the CLR it reports is "
                  "partial and must not be cited yet." % clr_name)
        else:
            clr_values[tag] = value
            if tag == "d=0":
                # d=0 is the leg the abstract quotes.
                expert_clr_pct = value

    # --- ablation A6: full-network fine-tune --------------------------------
    # Two artifacts: the 400-seed primary-protocol eval of the A6 weights, and
    # the 60-seed screening record that also carries the mask-permissive arm.
    a6_name = "ablation_A6_fullnet_d0_n400_s20000.json"
    a6_paired_name = "ablation_A6_vs_spc_paired_d0_n400.json"
    a6_train_name = "ablation_A6_fullnet_train.json"
    a6_pct = None
    if os.path.exists(os.path.join(EV, a6_name)):
        d = load(a6_name)
        a6_pct = 100.0 * float(d["kill_rate"])
        add("A6 full-network kill (400 seeds)", "%.2f%% (%d/%d)"
            % (a6_pct, d["termination_reasons"].get("target_killed", 0),
               d["episodes"]), a6_name)
        add("A6 full-network CLR", "%.2f%% (%d/%d)"
            % (100.0 * d["clr"], d["clr_fire_commands"], d["clr_allowed_steps"]),
            a6_name)
    else:
        add("A6 full-network kill (400 seeds)", "ABSENT (not run yet)", a6_name)
    a6_p = None
    if os.path.exists(os.path.join(EV, a6_paired_name)):
        d = load(a6_paired_name)
        a6_p = float(d["mcnemar"]["exact_two_sided_p"])
        add("A6 vs SPC paired delta", "%+.2f pp"
            % float(d["kill_rate"]["paired_diff_pp"]), a6_paired_name)
        add("A6 vs SPC discordant", str(d["contingency"]["discordant_total"]),
            a6_paired_name)
        add("A6 vs SPC exact p", "%.3e" % a6_p, a6_paired_name)
    a6_dev = None
    if os.path.exists(os.path.join(EV, a6_train_name)):
        d = load(a6_train_name)
        a6_dev = d["gates"]["hdg_spd_logits_max_diff"]
        add("A6 maneuver logit deviation", "%.2f (SPC: <1e-9)" % a6_dev,
            a6_train_name)
        add("A6 training mode", str(d.get("training_mode")), a6_train_name)

    # --- ablation A5: random-init fire head, encoder frozen ------------------
    a5_name = "ablation_A5_random_fire_head_train.json"
    if os.path.exists(os.path.join(EV, a5_name)):
        d = load(a5_name)
        mode = str(d.get("training_mode"))
        dev = d["gates"]["hdg_spd_logits_max_diff"]
        add("A5 training mode", mode, a5_name)
        add("A5 maneuver deviation", "%.2e (must be 0 in A5 mode)" % dev, a5_name)
        add("A5 val allowed-window acc", "%.4f" % d["best_val_allowed_acc"],
            a5_name)
        # In A5 the freeze pattern is SPC's, so the identity gates must still
        # pass. If the script ever reports a non-zero deviation here, the
        # ablation is no longer measuring what it claims to measure.
        if dev != 0.0:
            print("WARNING: A5 reports a non-zero maneuver deviation (%.3e). "
                  "A5 is supposed to keep SPC's freeze pattern, so this means "
                  "the ablation is not isolating initialisation." % dev)
    else:
        add("A5 random-init fire head", "ABSENT (in flight or not run)", a5_name)

    # --- ablation A5 at the primary protocol, paired against SPC ------------
    a5_p_name = "ablation_A5_random_fire_head_d0_n400_s20000.json"
    a5_vs_spc = "ablation_A5_vs_spc_paired_d0_n400.json"
    if os.path.exists(os.path.join(EV, a5_p_name)):
        d = load(a5_p_name)
        add("A5 kill (400 seeds)", "%.2f%% (%d/%d)"
            % (100.0 * d["kill_rate"],
               d["termination_reasons"].get("target_killed", 0), d["episodes"]),
            a5_p_name)
        add("A5 CLR (400 seeds)", "%.2f%% (%d/%d)"
            % (100.0 * d["clr"], d["clr_fire_commands"], d["clr_allowed_steps"]),
            a5_p_name)
    if os.path.exists(os.path.join(EV, a5_vs_spc)):
        d = load(a5_vs_spc)
        cont = d["contingency"]
        add("A5 vs SPC discordant", "%d (expect 0 -> behaviourally identical)"
            % cont["discordant_total"], a5_vs_spc)

    # --- environment-threshold sanity (should match Appendix B) -------------
    sys.path.insert(0, ROOT)
    from src.environment.singlecombat_shoot_task import (  # noqa: E402
        MIN_ATTACK_DISTANCE, MIN_ATTACK_INTERVAL, MAX_ATTACK_ANGLE, NUM_MISSILES)
    from scripts.generate_shoot_rule_expert import (  # noqa: E402
        ATA_STRICT, CLOSURE_MIN, DLZ_LO, DLZ_HI)
    add("env constants", "dist>=%g interval=%d ATA<%g missiles=%d"
        % (MIN_ATTACK_DISTANCE, MIN_ATTACK_INTERVAL, MAX_ATTACK_ANGLE, NUM_MISSILES),
        "singlecombat_shoot_task.py")
    add("expert gate", "ATA<%g closure<-%g DLZ in [%g,%g]"
        % (ATA_STRICT, CLOSURE_MIN, DLZ_LO, DLZ_HI), "generate_shoot_rule_expert.py")

    print("=" * 96)
    print("RECOMPUTED FROM LIVE ARTIFACTS (nothing taken from filenames)")
    print("=" * 96)
    for tag, val, art in rows:
        print("%-30s %-34s %s" % (tag, val, art))

    # cross-check: does the draft still contain the numbers we just computed?
    # An explicit path lets the build audit the PDF-extracted text instead of
    # the Markdown source; both go through exactly the same checks.
    audited = sys.argv[1] if len(sys.argv) > 1 else PAPER
    text = io.open(audited, encoding="utf-8").read()
    print()
    print("auditing: %s" % audited)
    checks = [
        ("46.50%", "E7 BC kill in text"),
        ("90.75%", "E7 SPC kill in text"),
        ("+44.25 pp", "E7 paired delta in text"),
        ("1.04e-53", "E7 p-value in text"),
        ("7.26%", "BC CLR in text"),
        ("42.25%", "E5 BC kill in text"),
        ("+48.50 pp", "E5 paired delta in text"),
        ("36.75%", "fresh-env expert d=0 in text"),
        ("25.75%", "fresh-env expert d=0.3 in text"),
        ("3.48e-05", "E3 d=0 p in text"),
        ("3.79e-11", "E3 d=0.3 p in text"),
        ("86.7%", "E1 asap in text"),
        ("15.0%", "E1 inherited in text"),
        ("7.32%", "Fig-2 seed CLR in text"),
        # Regression guard for the 2026-09-17 p-value error: this value came
        # from feeding TIE counts into exact_mcnemar. If it ever appears in
        # the draft, the mistake has been reintroduced.
        ("2.5e-65", "TRAP: must be ABSENT (ties passed as discordant)"),
        ("4.4e-65", "TRAP: must be ABSENT (same error, second instance)"),
    ]
    if expert_clr_pct is not None:
        # The CLR run is complete, so the draft must now carry its value and
        # must no longer carry any placeholder for it.
        checks.append(("%.2f%%" % expert_clr_pct, "E3-CLR d=0 expert CLR in text"))
        checks.append(("{{EXPERT_CLR",
                       "TRAP: must be ABSENT (placeholder left unfilled)"))
    else:
        checks.append(("{{EXPERT_CLR",
                       "placeholder still expected (CLR run incomplete)"))
    if "d=0.3" in clr_values:
        checks.append(("%.2f%%" % clr_values["d=0.3"],
                       "E3-CLR d=0.3 expert CLR in text"))
    if a6_pct is not None:
        checks.append(("%.2f%%" % a6_pct, "A6 kill rate in text"))
        checks.append(("99.67%", "A6 CLR in text"))
    if a6_p is not None:
        checks.append(("%.2e" % a6_p, "A6 vs SPC p-value in text"))
    if a6_dev is not None:
        checks.append(("%.2f" % a6_dev, "A6 maneuver deviation in text"))
    if os.path.exists(os.path.join(EV, a5_p_name)):
        checks.append(("99.94%", "A5 CLR (400 seeds) in text"))
        # Deliberately NOT a phrase check here. The A5 claim reads "zero
        # discordant seeds", and a phrase that long WILL straddle a column
        # break in a two-column PDF: pdftotext then interleaves the other
        # column between the words ("reports zero discordant" / <other column> /
        # "seeds"), in both -layout and reading-order mode. Phrase integrity is
        # therefore not reliably auditable from extracted PDF text, and a
        # checker that pretends otherwise fails on correct papers. The evidence
        # for this claim is numeric and is verified from the artifact above
        # ("A5 vs SPC discordant", which must read 0); the text check would only
        # re-assert the wording.

    # Both must-be-ABSENT traps are also *described* in the draft (Appendix D
    # and the section 9 notes), so a naive substring count reports a false
    # positive. Ignore hits on lines that are self-evidently talking about the
    # trap rather than asserting the number.
    TRAP_CONTEXT = ("bogus", "guard", "must be ABSENT", "must-be-absent",
                    "TRAP", "feeding tie counts", "placeholder")
    lines = text.splitlines()
    # Second view of the same text with all whitespace runs collapsed. In a
    # two-column PDF a phrase wraps at the column edge, so pdftotext yields
    # "zero discordant\n  seeds" and a literal needle search reports MISSING for
    # a phrase that is present and correct (hit on the first PDF build,
    # 2026-09-19). Value checks use the flat view; the line-based context logic
    # below still needs real lines.
    flat = re.sub(r"\s+", " ", text)

    def count_in_claim_context(needle):
        hits = 0
        for line in lines:
            if needle in line and not any(c in line for c in TRAP_CONTEXT):
                hits += 1
        return hits
    print()
    print("=" * 96)
    print("DRAFT TEXT OCCURRENCE CHECK")
    print("=" * 96)
    for needle, what in checks:
        hits = (count_in_claim_context(needle) if what.startswith("TRAP")
                else flat.count(needle))
        print("%-46s %s  (count=%d)" % (what, "FOUND" if hits else "MISSING", hits))
    print()
    print("Lines in draft:", len(text.splitlines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
