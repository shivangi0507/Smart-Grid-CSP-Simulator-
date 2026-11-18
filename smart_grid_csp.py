#!/usr/bin/env python3
"""
Smart Grid CSP Simulator with Kalman-filtered sensing (Python version)

Every 10 simulated minutes the grid agent
  1. reads its (noisy) sensors,
  2. cleans them with two Kalman filters,
  3. builds a Constraint Satisfaction Problem and solves it with
     backtracking + forward checking,
  4. applies the valid solution with the highest utility.

  Variables  : K blackout level, G gas level, P price, B battery command
  Domains    : K 0..4 | G 0..6 | P nine prices $0.10-$0.40 | B -4..+4 (- discharge, + charge)
  Constraints: unary, binary and higher-order (see build of `cons` in Grid.step)

Kalman filters (new in this version)
  * Renewable output  : 2-state (level, trend) filter, per-house units.
  * Battery charge    : 1-state filter that uses the agent's own battery command
                        as a control input (predict) and a noisy SoC meter (update).

Sensing modes (--sensing):
  kalman  noisy sensors -> Kalman filters -> controller   (default)
  raw     noisy sensors -> controller directly             (baseline)
  oracle  controller sees the true values                  (upper bound)

Usage:
  python smart_grid_csp.py                                  # 12 houses, sunny, 24 h, Kalman on
  python smart_grid_csp.py --houses 30 --weather heat --seed 2
  python smart_grid_csp.py --sensing raw --risk-k 0         # naive baseline: trust the raw sensors
  python smart_grid_csp.py --compare --seed 1               # raw vs kalman vs oracle table
  python smart_grid_csp.py --plot                           # matplotlib chart (optional)
"""
import argparse
import math
import random

# ---------------------------------------------------------------- constants
WEATHER = {  # sun factor, wind factor, temperature offset, label
    "sunny": (1.0, 0.25, 8, "Clear skies"),
    "cloudy": (0.3, 0.40, 0, "Overcast"),
    "windy": (0.7, 1.00, -2, "Strong wind"),
    "storm": (0.1, 0.70, -4, "Storm"),
    "heat": (1.0, 0.15, 19, "Heatwave"),
    "cold": (0.5, 0.35, -16, "Cold snap"),
}
BASE_PRICE, GAS_PRICE = 0.12, 0.30      # $/kWh
DT = 1 / 6                              # hours per tick (10 minutes)
CO2_PER_KWH = 0.45                      # kg CO2 per kWh of gas
MIN_SOC = 0.10
PRICES = [0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40]

# sensing / hidden-state parameters
REN_NOISE = 0.10      # renewable-output meter: 10 % (1 sigma) multiplicative error
SOC_NOISE = 0.03      # battery SoC meter: 3 percentage points (1 sigma)
RISK_K = 1.0          # plan against (estimate - RISK_K * sigma) of renewable output
BATT_EFF = 0.95       # real battery efficiency, one way. The agent does NOT know this.

# Kalman tuning (chosen on seeds 100-105, never on the seeds used in the report)
Q_LEVEL = 0.005       # process noise on renewable level, (kW per house)^2 per tick
Q_TREND = 0.0001      # process noise on renewable trend
Q_SOC = 1.0e-5        # process noise on SoC (covers the unmodelled efficiency loss)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fmt_time(h):
    t = int(round(h * 60)) % 1440
    return f"{t // 60:02d}:{t % 60:02d}"


# ---------------------------------------------------------------- Kalman filters
class LevelTrendKalman:
    """Constant-velocity Kalman filter for a slowly moving positive signal.

    State x = [level, trend], one tick = one time unit.
        predict : x <- F x,  P <- F P F' + Q      with F = [[1, 1], [0, 1]]
        update  : y = z - x[0];  S = P00 + R;  K = P[:,0] / S;  x += K y;  P <- (I - K H) P
    The sensor error is multiplicative, so R = (rel_sigma * level)^2.
    """

    def __init__(self, rel_sigma, q_level, q_trend, floor=0.05):
        self.rel, self.ql, self.qt, self.floor = rel_sigma, q_level, q_trend, floor
        self.x = None
        self.P = None
        self.last_gain = 0.0

    @property
    def sigma(self):
        """1-sigma uncertainty of the level estimate."""
        return math.sqrt(max(self.P[0][0], 0.0)) if self.P else 0.0

    def step(self, z):
        if self.x is None:                       # start from the first reading
            r0 = (self.rel * max(z, self.floor)) ** 2
            self.x = [z, 0.0]
            self.P = [[r0, 0.0], [0.0, 0.05]]
            return z
        (a, b), (c, d) = self.P
        # predict
        self.x = [self.x[0] + self.x[1], self.x[1]]
        self.P = [[a + b + c + d + self.ql, b + d],
                  [c + d, d + self.qt]]
        # update
        (a, b), (c, d) = self.P
        r = (self.rel * max(self.x[0], self.floor)) ** 2
        s = a + r
        k0, k1 = a / s, c / s
        y = z - self.x[0]
        self.x = [self.x[0] + k0 * y, self.x[1] + k1 * y]
        self.P = [[(1 - k0) * a, (1 - k0) * b],
                  [c - k1 * a, d - k1 * b]]
        self.last_gain = k0
        return max(0.0, self.x[0])


class ScalarKalman:
    """1-state Kalman filter with a control input (used for battery state of charge).

        predict : x <- x + u,  P <- P + Q       (u = what the agent *thinks* its command did)
        update  : K = P / (P + R);  x <- x + K (z - x);  P <- (1 - K) P
    """

    def __init__(self, q, r, x0, p0=1e-3):
        self.q, self.r, self.x, self.p = q, r, x0, p0
        self.last_gain = 0.0

    def step(self, z, u=0.0):
        self.x += u
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        self.last_gain = k
        return clamp(self.x, 0.0, 1.0)


# ---------------------------------------------------------------- CSP solver
class Constraint:
    def __init__(self, name, kind, scope, test):
        self.name, self.kind, self.scope, self.test = name, kind, scope, test


def solve_csp(domains, constraints):
    """Backtracking search with forward checking and smallest-domain-first
    variable ordering. Returns (all_solutions, stats)."""
    stats = {"nodes": 0, "backtracks": 0, "dwo": 0}
    solutions = []
    names = list(domains)

    def bt(cur, asg):
        unassigned = [v for v in names if v not in asg]
        if not unassigned:
            solutions.append(dict(asg))
            return
        v = min(unassigned, key=lambda x: len(cur[x]))        # MRV heuristic
        for d in cur[v]:
            stats["nodes"] += 1
            asg[v] = d
            nxt = dict(cur)
            wipe = False
            for c in constraints:
                if v not in c.scope:
                    continue
                free = [x for x in c.scope if x not in asg]
                if not free:
                    if not c.test(asg):
                        wipe = True
                        break
                elif len(free) == 1:                          # forward check
                    x = free[0]
                    kept = []
                    for y in nxt[x]:
                        asg[x] = y
                        if c.test(asg):
                            kept.append(y)
                        del asg[x]
                    nxt[x] = kept
                    if not kept:                              # domain wipe-out
                        wipe = True
                        stats["dwo"] += 1
                        break
            if wipe:
                stats["backtracks"] += 1
            else:
                bt(nxt, asg)
            del asg[v]

    bt(domains, {})
    return solutions, stats


# ---------------------------------------------------------------- grid agent
class Grid:
    def __init__(self, houses, weather, seed=None, sensing="kalman",
                 ren_noise=REN_NOISE, soc_noise=SOC_NOISE,
                 q_level=Q_LEVEL, q_trend=Q_TREND, q_soc=Q_SOC, risk_k=RISK_K):
        assert sensing in ("kalman", "raw", "oracle")
        self.n, self.weather, self.sensing = houses, weather, sensing
        self.risk_k = risk_k
        self.rng = random.Random(seed)                                  # environment
        self.srng = random.Random(None if seed is None else seed + 7919)  # sensors (separate stream)
        self.ren_noise = 0.0 if sensing == "oracle" else ren_noise
        self.soc_noise = 0.0 if sensing == "oracle" else soc_noise
        self.hour = 6.0
        self.soc = 0.5                 # TRUE battery state of charge (hidden from the agent)
        self.dir, self.since, self.cycles = 0, 9, 0
        self.price = BASE_PRICE
        self.cloud, self.gust = 1.0, 1.0   # hidden weather drift
        self.fossil_saved = self.money_saved = self.bill = self.gas_kwh = 0.0
        self.unsat_ticks = 0
        self.soc_breaches = 0
        self.history = []
        # Kalman filters
        self.kf_ren = LevelTrendKalman(max(self.ren_noise, 1e-3), q_level, q_trend)
        self.kf_soc = ScalarKalman(q_soc, max(self.soc_noise, 1e-3) ** 2, 0.5)
        self.u_prev = 0.0              # nominal change in SoC caused by last command
        self.soc_margin = 0.0          # safety buffer above MIN_SOC the agent keeps (set in sense)
        self.ren_sigma = 0.0           # 1-sigma uncertainty of the renewable estimate (kW, set in sense)

    # --- environment model -------------------------------------------------
    def environment(self):
        sun, wind_f, t_off, _ = WEATHER[self.weather]
        h = self.hour % 24
        self.cloud = clamp(self.cloud + (self.rng.random() - .5) * .25, .55, 1)
        self.gust = clamp(self.gust + (self.rng.random() - .5) * .3, .5, 1.3)
        temp = t_off + 14 + 6 * math.sin(math.pi * (h - 9) / 12) + (6 if t_off > 10 else 0)
        solar = self.n * 2.2 * max(0, math.sin(math.pi * (h - 6) / 12)) * sun * (1 if sun < .5 else self.cloud)
        wind = self.n * wind_f * self.gust
        peak = math.exp(-((h - 8) ** 2) / 4) + 1.3 * math.exp(-((h - 19.5) ** 2) / 5)
        base = .35 + .45 * peak
        flex = (max(0, temp - 26) * .09 + max(0, 12 - temp) * .07 + (.5 if 18 <= h < 24 else 0))
        return temp, solar, wind, base, flex

    # --- sensing -----------------------------------------------------------
    def sense(self, ren_true):
        """Return (measured renewable, estimated renewable, measured SoC, estimated SoC)."""
        n = self.n
        z_ren = max(0.0, ren_true * (1 + self.srng.gauss(0, self.ren_noise)))
        z_soc = self.soc + self.srng.gauss(0, self.soc_noise)
        if self.sensing == "kalman":
            ren_hat = n * self.kf_ren.step(z_ren / n)
            self.ren_sigma = n * self.kf_ren.sigma
            soc_hat = self.kf_soc.step(z_soc, self.u_prev)
            # the filter knows how unsure it is: keep a 2-sigma buffer above the floor
            self.soc_margin = 2 * math.sqrt(self.kf_soc.p + self.kf_soc.q)
        elif self.sensing == "raw":
            ren_hat, soc_hat = z_ren, clamp(z_soc, 0.0, 1.0)
            self.ren_sigma = self.ren_noise * z_ren          # rated meter accuracy
            self.soc_margin = 2 * self.soc_noise            # best it can do: trust the meter's rated noise
        else:  # oracle
            ren_hat, soc_hat = ren_true, self.soc
            self.ren_sigma, self.soc_margin = 0.0, 0.0
        return z_ren, ren_hat, z_soc, soc_hat

    # --- one control step --------------------------------------------------
    def step(self):
        n = self.n
        temp, solar, wind, base, flex = self.environment()
        ren = solar + wind                                         # true output
        z_ren, obs, z_soc, soc_hat = self.sense(ren)               # what the agent works with
        margin = self.soc_margin
        ren_est = obs
        obs = max(0.0, obs - self.risk_k * self.ren_sigma)         # plan against the pessimistic value

        def demand(p):
            return n * (base + flex * (1 - clamp((p - .15) / .25, 0, 1) * .65))

        extreme = temp > 34 or temp < 2
        cap = 0.26 if extreme else 0.40                           # regulatory cap
        cap_kwh, step_b, step_g = n * 6, n * .4, n * 1.4 / 6

        def served(a):  return demand(a["P"]) * (1 - .04 * a["K"])
        def imbalance(a): return obs + a["G"] * step_g - a["B"] * step_b - served(a)

        def attempt(relax):
            tol = .08 if relax else .04
            m = lambda a: tol * served(a)
            cons = [
                Constraint("price <= regulatory cap", "unary", ["P"], lambda a: a["P"] <= cap),
                Constraint("battery energy limits", "unary", ["B"],
                           lambda a: (a["B"] * step_b * DT <= (1 - soc_hat) * cap_kwh) if a["B"] > 0 else
                                     (-a["B"] * step_b * DT <= (soc_hat - MIN_SOC - margin) * cap_kwh) if a["B"] < 0 else True),
                Constraint("no rapid battery flip", "unary", ["B"],
                           lambda a: not (self.dir and self.since < 3) or (a["B"] > 0) - (a["B"] < 0) != -self.dir),
                Constraint("no blackout in extreme weather", "unary", ["K"],
                           lambda a: relax or not extreme or a["K"] == 0),
                Constraint("no charging from gas", "binary", ["G", "B"], lambda a: not (a["G"] > 0 and a["B"] > 0)),
                Constraint("blackout only if gas maxed", "binary", ["K", "G"], lambda a: a["K"] == 0 or a["G"] == 6),
                Constraint("blackout only if price at cap", "binary", ["K", "P"], lambda a: a["K"] == 0 or a["P"] == cap),
                Constraint("frequency band", "higher-order", ["K", "G", "P", "B"],
                           lambda a: imbalance(a) >= -m(a)),
                Constraint("no wasted gas/battery/blackout", "higher-order", ["K", "G", "P", "B"],
                           lambda a: (a["G"] == 0 or imbalance(a) - step_g < -m(a)) and
                                     (a["B"] >= 0 or imbalance(a) - step_b < -m(a)) and
                                     (a["K"] == 0 or imbalance(a) - .04 * demand(a["P"]) < -m(a))),
            ]
            b_star = clamp(round((obs - demand(self.price)) / step_b), -4, 4)
            doms = {"K": list(range(5)), "G": list(range(7)), "P": list(PRICES),
                    "B": sorted(range(-4, 5), key=lambda b: abs(b - b_star))}   # value ordering
            for c in cons:
                if len(c.scope) == 1:
                    v = c.scope[0]
                    doms[v] = [d for d in doms[v] if c.test({v: d})]
            sols, st = solve_csp(doms, [c for c in cons if len(c.scope) > 1])
            return sols, st, relax

        sols, stats, relaxed = attempt(False)
        if not sols:
            self.unsat_ticks += 1
            sols, stats, relaxed = attempt(True)

        def utility(a):
            s = served(a)
            return ((obs - max(0, imbalance(a))) - 3 * a["G"] * step_g
                    - 8 * (demand(a["P"]) - s) - .3 * abs(a["B"]) * step_b
                    - .4 * s * (a["P"] - BASE_PRICE) / .28)

        best = max(sols, key=utility) if sols else {"K": 4, "G": 6, "P": cap, "B": 0}

        # ---- the real world reacts (uses TRUE values, not the agent's estimates)
        bat_kw, gas = best["B"] * step_b, best["G"] * step_g
        actual = demand(best["P"]) * (1 + (self.rng.random() - .5) * .03)
        srv = actual * (1 - .04 * best["K"])
        imb = ren + gas - bat_kw - srv
        freq = clamp(60 + 1.25 * min(imb, 0) / srv, 59.9, 60.1)
        soc_before = self.soc
        eff = BATT_EFF if bat_kw > 0 else 1 / BATT_EFF            # real losses, unknown to the agent
        self.soc = clamp(self.soc + bat_kw * eff * DT / cap_kwh, 0, 1)
        if self.soc < MIN_SOC - 1e-6:
            self.soc_breaches += 1
        self.u_prev = bat_kw * DT / cap_kwh                       # agent's own (lossless) model
        self.price = best["P"]
        sg = (best["B"] > 0) - (best["B"] < 0)
        if sg and sg != self.dir:
            if self.dir:
                self.cycles += 1
            self.dir, self.since = sg, 0
        else:
            self.since += 1
        self.gas_kwh += gas * DT
        self.fossil_saved += max(0, srv - gas) * DT
        self.money_saved += srv * DT * (GAS_PRICE - self.price)
        self.bill += srv * DT * self.price
        rec = dict(time=fmt_time(self.hour), temp=temp, demand=actual, ren=ren, gas=gas,
                   battery=bat_kw, soc=self.soc, price=self.price, freq=freq,
                   blackout=actual * .04 * best["K"], solutions=len(sols),
                   nodes=stats["nodes"], backtracks=stats["backtracks"], dwo=stats["dwo"], relaxed=relaxed,
                   ren_meas=z_ren, ren_est=ren_est, ren_plan=obs, ren_sigma=self.ren_sigma, soc_meas=z_soc, soc_est=soc_hat, soc_true=soc_before)
        self.history.append(rec)
        self.hour += DT
        return rec


# ---------------------------------------------------------------- experiment helper
def run_scenario(houses, weather, hours, seed, sensing="kalman", **kw):
    """Run one simulation and return a dictionary of summary metrics."""
    g = Grid(houses, weather, seed, sensing, **kw)
    for _ in range(hours * 6):
        g.step()
    h = g.history
    nT = len(h)
    rmse = lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / nT)
    ren_t = [r["ren"] for r in h]
    return {
        "fossil_saved": g.fossil_saved, "co2": g.fossil_saved * CO2_PER_KWH, "gas_kwh": g.gas_kwh,
        "bill": g.bill, "money_saved": g.money_saved, "cycles": g.cycles, "unsat": g.unsat_ticks,
        "out_of_band": sum(abs(r["freq"] - 60) > .05 for r in h), "ticks": nT,
        "soc_breaches": g.soc_breaches,
        "min_soc": min(r["soc"] for r in h),
        "ren_rmse_pct": 100 * rmse([r["ren_est"] for r in h], ren_t) / (sum(ren_t) / nT),
        "soc_rmse_pts": 100 * rmse([r["soc_est"] for r in h], [r["soc_true"] for r in h]),
        "nodes_min": min(r["nodes"] for r in h), "nodes_max": max(r["nodes"] for r in h),
        "sols_min": min(r["solutions"] for r in h), "sols_max": max(r["solutions"] for r in h),
        "ms_per_step": 0.0, "grid": g,
    }


# ---------------------------------------------------------------- main
def main(args_list=None):
    ap = argparse.ArgumentParser(description="Smart Grid CSP Simulator with Kalman filtering")
    ap.add_argument("--houses", type=int, default=12)
    ap.add_argument("--weather", choices=WEATHER, default="sunny")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sensing", choices=["kalman", "raw", "oracle"], default="kalman")
    ap.add_argument("--ren-noise", type=float, default=REN_NOISE, help="renewable meter noise (fraction, 1 sigma)")
    ap.add_argument("--soc-noise", type=float, default=SOC_NOISE, help="SoC meter noise (fraction of capacity)")
    ap.add_argument("--risk-k", type=float, default=RISK_K, help="plan against estimate - k*sigma (0 = trust the estimate)")
    ap.add_argument("--compare", action="store_true", help="run raw, kalman and oracle and print a table")
    ap.add_argument("--verbose", action="store_true", help="print every tick")
    ap.add_argument("--plot", action="store_true", help="show a matplotlib chart (optional)")
    args = ap.parse_args(args_list)
    kw = dict(ren_noise=args.ren_noise, soc_noise=args.soc_noise, risk_k=args.risk_k)

    if args.compare:
        print(f"{args.houses} houses, {WEATHER[args.weather][3]}, {args.hours} h, seed {args.seed}\n")
        rows = [("Fossil fuel avoided (kWh)", "fossil_saved", "{:,.0f}"), ("Gas burned (kWh)", "gas_kwh", "{:,.0f}"),
                ("Out-of-band ticks", "out_of_band", "{}"), ("Unsatisfiable ticks", "unsat", "{}"),
                ("Battery direction switches", "cycles", "{}"), ("SoC floor breaches", "soc_breaches", "{}"),
                ("Renewable est. error (% RMSE)", "ren_rmse_pct", "{:.1f}"),
                ("SoC est. error (pts RMSE)", "soc_rmse_pts", "{:.2f}")]
        res = {m: run_scenario(args.houses, args.weather, args.hours, args.seed, m, **kw)
               for m in ("oracle", "raw", "kalman")}
        print(f"{'':32}{'oracle':>10}{'raw':>10}{'kalman':>10}")
        for label, key, f in rows:
            print(f"{label:32}" + "".join(f"{f.format(res[m][key]):>10}" for m in res))
        return

    grid = Grid(args.houses, args.weather, args.seed, args.sensing, **kw)
    print(f"Smart grid: {args.houses} houses, {WEATHER[args.weather][3]}, {args.hours} h, sensing = {args.sensing}\n")
    print(f"{'time':>5} {'temp':>5} {'demand':>8} {'renew':>8} {'meas':>8} {'est':>8} {'gas':>7} {'batt':>7} "
          f"{'SoC':>5} {'price':>6} {'Hz':>6} {'sols':>5}")
    for i in range(args.hours * 6):
        r = grid.step()
        if args.verbose or i % 6 == 0:   # hourly rows unless --verbose
            print(f"{r['time']:>5} {r['temp']:>4.0f}C {r['demand']:>7.1f}k {r['ren']:>7.1f}k {r['ren_meas']:>7.1f}k "
                  f"{r['ren_est']:>7.1f}k {r['gas']:>6.1f}k {r['battery']:>+6.1f}k {r['soc']*100:>4.0f}% "
                  f"${r['price']:>.2f} {r['freq']:>6.2f} {r['solutions']:>5}" + ("  RELAXED" if r["relaxed"] else ""))

    h = grid.history
    out_of_band = sum(abs(r["freq"] - 60) > .05 for r in h)
    nT = len(h)
    ren_t = [r["ren"] for r in h]
    rm = lambda k, ref: math.sqrt(sum((r[k] - r[ref]) ** 2 for r in h) / nT)
    print("\n--- Summary ---")
    print(f"Fossil fuel avoided : {grid.fossil_saved:,.0f} kWh  (~{grid.fossil_saved * CO2_PER_KWH:,.0f} kg CO2)")
    print(f"Gas burned          : {grid.gas_kwh:,.0f} kWh")
    print(f"City bill           : ${grid.bill:,.0f}  (saved ${grid.money_saved:,.0f} vs gas-only)")
    print(f"Battery switches    : {grid.cycles}")
    print(f"Unsatisfiable ticks : {grid.unsat_ticks}")
    print(f"Out-of-band freq    : {out_of_band} of {nT} ticks")
    print(f"SoC floor breaches  : {grid.soc_breaches} ticks below {MIN_SOC:.0%}")
    print(f"Renewable estimate  : {100 * rm('ren_est', 'ren') / (sum(ren_t) / nT):.1f} % RMSE "
          f"(raw sensor: {100 * rm('ren_meas', 'ren') / (sum(ren_t) / nT):.1f} %)")
    print(f"SoC estimate        : {100 * rm('soc_est', 'soc_true'):.2f} pts RMSE "
          f"(raw meter: {100 * rm('soc_meas', 'soc_true'):.2f} pts)")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Install matplotlib to use --plot:  pip install matplotlib")
            return
        t = [i * DT for i in range(nT)]
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(t, [r["ren_meas"] for r in h], color="lightgray", label="Noisy sensor")
        ax[0].plot(t, ren_t, color="tab:green", label="True")
        ax[0].plot(t, [r["ren_est"] for r in h], color="tab:blue", ls="--", label="Estimate used by agent")
        ax[0].set_ylabel("Renewable kW"); ax[0].legend()
        ax[1].plot(t, [r["soc_meas"] * 100 for r in h], color="lightgray", label="SoC meter")
        ax[1].plot(t, [r["soc_true"] * 100 for r in h], color="tab:green", label="True SoC")
        ax[1].plot(t, [r["soc_est"] * 100 for r in h], color="tab:blue", ls="--", label="Estimate")
        ax[1].set_ylabel("SoC %"); ax[1].legend()
        ax[2].plot(t, [r["demand"] for r in h], color="tab:red", label="Demand")
        ax[2].plot(t, [r["gas"] for r in h], color="tab:orange", label="Gas")
        ax[2].set_ylabel("kW"); ax[2].set_xlabel("Hours since 06:00"); ax[2].legend()
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
