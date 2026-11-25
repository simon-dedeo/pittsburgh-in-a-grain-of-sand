#!/usr/bin/env python3
"""Reproduce all v12 figures and summary statistics directly from ./data/.

Regenerates:
  - crf_and_m2.pdf       (Fig. 2: empirical CRF exposure-ratio calibration & Stan M2 scatter forcing)
  - profile_v12.pdf      (Fig. 3: 2D multi-bracket transmittance map & radial profile)
  - ab_longitudinal.pdf  (Fig. 5: 4-month prospective A/B verification + 2026-11-24 bayes3 upgrade)
  - creep_4month.pdf     (Fig. 7: 125-day trajectory & Andrade t^(1/3) viscoelastic creep law)

Usage:
    python3 reproduce_figures.py                         # writes PDFs to ./reproduced_figs/
    python3 reproduce_figures.py --out /some/dir         # custom output directory
    python3 reproduce_figures.py --overwrite-paper-figs  # update ./paper/figs/ in place
    python3 reproduce_figures.py --run-stan              # also refit spot_m2.stan via cmdstanpy
"""
import argparse
import collections
import csv
import datetime
import gzip
import json
import os
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def crf_eval(v, a, g):
    x = np.clip(np.asarray(v, np.float64) / 255.0, 0.0, 1.0)
    c0 = (a / (1.0 + a)) ** g
    return (((x + a) / (1.0 + a)) ** g - c0) / (1.0 - c0)


def srgb_lin(v):
    x = np.clip(np.asarray(v, np.float64) / 255.0, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_photometry_gz(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            if r.get("patch") != "spot" or r.get("gain") != "0":
                continue
            try:
                bg, dk, fac = float(r["bg_fit"]), float(r["disk_med"]), float(r["facade_med"])
                exp, mad = float(r["exp_units"]), float(r["ann_resid_mad"])
                if float(r["clip_hi"]) < 0.005 and float(r["clip_lo"]) < 0.005 and mad < 2.0 and exp > 0 and 15 < bg < 248:
                    rows.append((r["ts_utc"], r["role"], r["ch"], bg, dk, fac, exp, mad))
            except (KeyError, ValueError, TypeError):
                pass
    return rows


# ------------------------------------------------------------------ Fig. 2: CRF & Stan M2
def make_crf_and_m2(out_dir, run_stan=False):
    clean = load_photometry_gz(DATA / "spot_photometry_14d.csv.gz")
    bayes = json.loads((DATA / "bayes_fit.json").read_text())
    model = json.loads((DATA / "spot_model.json").read_text())

    # Fit CRF on same-tick bracket pairs
    by_tick = collections.defaultdict(dict)
    for ts, role, ch, bg, dk, fac, exp, mad in clean:
        if role in ("br0", "br1", "br2") and mad < 1.5:
            by_tick[(ts[:16], ch)][role] = (bg, exp)

    crf_fit = {}
    err_srgb, err_crf = {}, {}
    default_crf = {"b": [0.55, 3.8], "g": [0.45, 3.8], "r": [0.36, 3.2]}
    for ch in "bgr":
        pairs = []
        for (_, c), d in by_tick.items():
            if c != ch:
                continue
            for k0, k1 in (("br0", "br1"), ("br1", "br2"), ("br0", "br2")):
                if k0 in d and k1 in d and d[k1][1] > d[k0][1] * 1.2 and d[k1][0] > d[k0][0] + 5:
                    pairs.append((d[k0][0], d[k1][0], d[k1][1] / d[k0][1]))
        arr = np.array(pairs)
        def loss(par):
            a, g = par
            if not (0.0 <= a <= 1.5 and 1.2 <= g <= 4.5):
                return 1e9
            r = np.log((crf_eval(arr[:, 1], a, g) / np.maximum(crf_eval(arr[:, 0], a, g), 1e-9)) / arr[:, 2])
            return float(np.mean(np.where(np.abs(r) < 0.05, 0.5 * r * r, 0.05 * (np.abs(r) - 0.025))))
        res = minimize(loss, default_crf[ch], method="Nelder-Mead")
        a, g = float(res.x[0]), float(res.x[1])
        crf_fit[ch] = (a, g)
        err_srgb[ch] = float(np.median(np.abs(np.log((srgb_lin(arr[:, 1]) / np.maximum(srgb_lin(arr[:, 0]), 1e-9)) / arr[:, 2])))) * 100
        err_crf[ch] = float(np.median(np.abs(np.log((crf_eval(arr[:, 1], a, g) / np.maximum(crf_eval(arr[:, 0], a, g), 1e-9)) / arr[:, 2])))) * 100

    print("\n=== Empirical CRF Calibration (Fig. 2 left) ===")
    for ch in "bgr":
        a, g = crf_fit[ch]
        print(f"  {ch.upper()}: a={a:.4f}, gamma={g:.3f}  |  median exposure-ratio err: sRGB {err_srgb[ch]:.1f}% -> CRF {err_crf[ch]:.1f}%")

    # Optional full Stan refit
    if run_stan:
        import cmdstanpy
        cutoff = max(r[0] for r in clean)[:10]
        dt_end = datetime.datetime.fromisoformat(max(r[0] for r in clean))
        cut7 = (dt_end - datetime.timedelta(days=7)).isoformat()
        ch_idx = {"b": 1, "g": 2, "r": 3}
        data_rows = []
        for ts, role, ch, bg, dk, fac, exp, mad in clean:
            if ts < cut7 or role not in ("br0", "br1", "br2") or mad >= 1.5 or not (20 < bg < 245):
                continue
            a, g = crf_fit[ch]
            Lbg = float(crf_eval(bg, a, g))
            Ldk = float(crf_eval(dk, a, g))
            Lfac = float(crf_eval(fac, a, g))
            if Lbg > 0.015:
                y = Ldk / Lbg
                if 0.85 < y < 1.05:
                    sig = max(0.002, (mad / max(bg, 20.0)) * g * 0.25)
                    data_rows.append((ch_idx[ch], y, min(Lfac / Lbg, 8.0), sig))
        if len(data_rows) > 3000:
            step = len(data_rows) / 3000.0
            data_rows = [data_rows[int(i * step)] for i in range(3000)]
        stan_data = dict(
            N=len(data_rows),
            ch=[d[0] for d in data_rows],
            y=[d[1] for d in data_rows],
            x_fac=[d[2] for d in data_rows],
            sig_m=[d[3] for d in data_rows],
        )
        m = cmdstanpy.CmdStanModel(stan_file=str(ROOT / "production/akdeniz/spot_m2.stan"))
        fit = m.sample(data=stan_data, chains=4, parallel_chains=4, iter_warmup=500, iter_sampling=500,
                       seed=42, show_progress=False)
        summ = fit.summary()
        print(f"\n=== Stan M2 Refit (N={len(data_rows)}, max Rhat={summ['R_hat'].max():.4f}) ===")
        for i, ch in enumerate("bgr", 1):
            print(f"  {ch.upper()}: t={summ.loc[f't[{i}]','Mean']:.5f}, k_fac={summ.loc[f'k_fac[{i}]','Mean']:+.5f}, "
                  f"sig_x={summ.loc[f'sig_x[{i}]','Mean']:.5f}, nu={summ.loc[f'nu[{i}]','Mean']:.2f}")

    # Plot Fig. 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200)
    colors = {"b": "#1f77b4", "g": "#2ca02c", "r": "#d62728"}
    labels = {"b": "Blue", "g": "Green", "r": "Red"}

    # Left panel: CRF curves vs sRGB
    v = np.linspace(0, 255, 256)
    ax1.plot(v, srgb_lin(v), "k--", lw=1.4, label="Nominal sRGB")
    for ch in "bgr":
        a, g = model["crf"][ch]
        ax1.plot(v, crf_eval(v, a, g), color=colors[ch], lw=1.8,
                 label=f"{labels[ch]} CRF ($a={a:.2f}, \\gamma={g:.2f}$; err {err_srgb[ch]:.0f}%→{err_crf[ch]:.1f}%)")
    ax1.set_xlabel("Encoded pixel value $v$ (0–255)")
    ax1.set_ylabel("Relative linear radiance $L(v)$")
    ax1.set_title("Empirical Camera Response Function")
    ax1.legend(frameon=False, loc="upper left", fontsize=7.6)
    ax1.grid(alpha=0.25)

    # Right panel: M2 scatter forcing (L_spot/L_bg vs L_fac/L_bg)
    dt_end = datetime.datetime.fromisoformat(max(r[0] for r in clean))
    cut7 = (dt_end - datetime.timedelta(days=7)).isoformat()
    print("\n=== Live Stan M2 Posterior (Table 1, from data/bayes_fit.json) ===")
    print(f"  N={bayes['n']}, max Rhat={bayes['rhat_max']:.4f}, diag_ok={bayes['diag_ok']}")
    for ch in "bgr":
        a, g = model["crf"][ch]
        xs, ys = [], []
        for ts, role, c, bg, dk, fac, exp, mad in clean:
            if c != ch or ts < cut7 or role not in ("br0", "br1", "br2") or mad >= 1.5 or not (20 < bg < 245):
                continue
            Lbg, Ldk, Lfac = float(crf_eval(bg, a, g)), float(crf_eval(dk, a, g)), float(crf_eval(fac, a, g))
            if Lbg > 0.015:
                y = Ldk / Lbg
                xf = Lfac / Lbg
                if 0.88 < y < 1.03 and 0.0 <= xf <= 3.5:
                    xs.append(xf)
                    ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        # Binned medians
        bins = np.linspace(0, 3.0, 11)
        bx, by, berr = [], [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (xs >= b0) & (xs < b1)
            if m.sum() >= 8:
                bx.append(np.median(xs[m]))
                by.append(np.median(ys[m]))
                berr.append(1.4826 * np.median(np.abs(ys[m] - np.median(ys[m]))) / np.sqrt(m.sum()))
        ax2.errorbar(bx, by, yerr=berr, fmt="o", ms=4.5, color=colors[ch], alpha=0.85, capsize=2)
        cinfo = bayes["channels"][ch]
        t_hat, k_hat = cinfo["t"], cinfo["k_fac"]
        xgrid = np.linspace(0, 3.0, 100)
        ax2.plot(xgrid, t_hat + k_hat * xgrid, color=colors[ch], lw=1.8,
                 label=f"{labels[ch]}: $t={t_hat:.4f}$, $k_{{\\mathrm{{fac}}}}={k_hat:+.4f}$")
        print(f"  {ch.upper()}: t={t_hat:.5f} {cinfo['t_ci']}, k_fac={k_hat:+.5f} {cinfo['k_fac_ci']}, "
              f"sig_x={cinfo['sig_x']:.5f}, nu={cinfo['nu']:.2f}")

    ax2.set_xlabel("Direct-sun forcing $L_{\\mathrm{fac}} / L_{\\mathrm{bg}}$")
    ax2.set_ylabel("Disk-to-sky transmittance $L_{\\mathrm{spot}} / L_{\\mathrm{bg}}$")
    ax2.set_title(f"Stan M2 Forward-Scattering Fit ($N={bayes['n']}$, $\\hat{{R}}={bayes['rhat_max']:.4f}$)")
    ax2.set_xlim(-0.05, 3.1)
    ax2.set_ylim(0.940, 0.978)
    ax2.legend(frameon=False, loc="upper left", fontsize=7.8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    p = out_dir / "crf_and_m2.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ Fig. 3: 2D TMAP & Radial Profile
def make_profile_v12(out_dir):
    T = np.load(DATA / "spot_tmap.npy").astype(np.float64)
    model = json.loads((DATA / "spot_model.json").read_text())
    tmm = model["tmap"]
    mx, my = float(tmm["mx"]), float(tmm["my"])
    dip = (1.0 - T) * 100.0  # percent dip, BGR
    dip_g = dip[:, :, 1]

    # Measure FWHM & second-moment ellipticity on half-max footprint
    yy, xx = np.mgrid[0:T.shape[0], 0:T.shape[1]]
    rr = np.hypot(xx - mx, yy - my)
    core_peak = float(np.percentile(dip_g[rr < 25], 95))
    mask = (dip_g >= 0.5 * core_peak) & (rr < 75) & (yy >= 20)
    dx = xx[mask] - mx
    dy = yy[mask] - my
    cov = np.cov(np.vstack([dx, dy]))
    evals, evecs = np.linalg.eigh(cov)
    ellip = float(np.sqrt(evals[1] / evals[0]))
    major_vec = evecs[:, 1]
    pa_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])) % 180.0)
    eq_diam = float(2.0 * np.sqrt(mask.sum() / np.pi))

    # Radial direction from image center (1920, 1080) to spot (x0+mx, y0+my)
    cx_full = tmm["x0"] + mx
    cy_full = tmm["y0"] + my
    rad_ang = float(np.degrees(np.arctan2(cy_full - 1080.0, cx_full - 1920.0)) % 180.0)
    ang_diff = min(abs(pa_deg - rad_ang), 180.0 - abs(pa_deg - rad_ang))

    print("\n=== 2D Transmittance Map Geometry (Fig. 3) ===")
    print(f"  Core depth BGR: B={tmm['core_depth']['b']*100:.2f}%, G={tmm['core_depth']['g']*100:.2f}%, R={tmm['core_depth']['r']*100:.2f}%")
    print(f"  Half-max equiv. diameter (FWHM): {eq_diam:.1f} px; ellipticity: {ellip:.3f}")
    print(f"  Major axis PA: {pa_deg:.1f} deg vs. radial from image center: {rad_ang:.1f} deg (offset {ang_diff:.1f} deg)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=200)
    sub = dip_g[int(my - 65):int(my + 95), int(mx - 80):int(mx + 80)]
    im = ax1.imshow(sub, origin="upper", cmap="magma", vmin=0, vmax=6.5,
                    extent=[-80, 80, 95, -65])
    ax1.axhline(20 - my, color="cyan", ls="--", lw=1.0, label="Top mask ($y=20$)")
    ax1.contour(np.arange(-80, 80), np.arange(-65, 95), sub, levels=[0.5 * core_peak],
                colors="white", linewidths=1.2)
    cb = fig.colorbar(im, ax=ax1, shrink=0.88, pad=0.02)
    cb.set_label("Green linear dip (%)")
    ax1.set_xlabel("$\\Delta x$ from centre (px)")
    ax1.set_ylabel("$\\Delta y$ from centre (px)")
    ax1.set_title(f"Registered 2D Shadow (FWHM={eq_diam:.0f} px, ellip={ellip:.3f})")
    ax1.legend(frameon=False, loc="lower right", fontsize=7.8)

    # Radial profile per channel
    rbins = np.arange(0, 105, 3)
    rc = 0.5 * (rbins[:-1] + rbins[1:])
    for c, (ch, col, lbl) in enumerate([("b", "#1f77b4", "Blue"), ("g", "#2ca02c", "Green"), ("r", "#d62728", "Red")]):
        prof = []
        for r0, r1 in zip(rbins[:-1], rbins[1:]):
            m = (rr >= r0) & (rr < r1) & (yy >= 20)
            prof.append(float(np.median(dip[:, :, c][m])) if m.sum() >= 10 else np.nan)
        ax2.plot(rc, prof, color=col, lw=1.9, label=f"{lbl} (n={tmm['n_ch'][c]})")
    ax2.axhline(0, color="gray", lw=0.8, ls=":")
    ax2.axvline(eq_diam / 2.0, color="k", lw=0.9, ls="--", alpha=0.6, label=f"Half-max radius ({eq_diam/2:.1f} px)")
    ax2.set_xlabel("Radius from centre $r$ (px)")
    ax2.set_ylabel("Linear dip $1 - T(r)$ (%)")
    ax2.set_title("Multi-Bracket Empirical-CRF Radial Profile")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.5, 6.4)
    ax2.legend(frameon=False, loc="upper right", fontsize=8.0)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    p = out_dir / "profile_v12.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ Fig. 5: Prospective A/B
def _read_ab(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                dt = datetime.datetime.strptime(r["stamp"][:15], "%Y%m%d_%H%M%S")
                clean = int(r["clean"])
                lvl = float(r["level"])
                bef = float(r["before"])
                aft = float(r["after"])
                mb = float(r["m_before"]) if "m_before" in r and r["m_before"] != "" else None
                ma = float(r["m_after"]) if "m_after" in r and r["m_after"] != "" else None
                ab = float(r["after_b"]) if "after_b" in r and r["after_b"] != "" else None
                ag = float(r["after_g"]) if "after_g" in r and r["after_g"] != "" else None
                ar = float(r["after_r"]) if "after_r" in r and r["after_r"] != "" else None
                rows.append(dict(dt=dt, clean=clean, lvl=lvl, bef=bef, aft=aft, mb=mb, ma=ma, ab=ab, ag=ag, ar=ar))
            except (KeyError, ValueError):
                pass
    return rows


def make_ab_longitudinal(out_dir):
    v1 = _read_ab(DATA / "ab_log_v1.csv")
    v2 = _read_ab(DATA / "ab_log_v2.csv")
    b3 = _read_ab(DATA / "ab_verify_20261124_bayes3.csv")
    sw = []
    with open(DATA / "sweep2.csv") as fh:
        for r in csv.DictReader(fh):
            try:
                if float(r.get("ring_rough", 0)) <= 0.04 or int(r.get("clean", 1)) == 1:
                    sw.append(r)
            except ValueError:
                pass

    def _summ(name, rows, use_mertens=False):
        cl = [r for r in rows if r["clean"] == 1 and r["lvl"] >= 50]
        kb, ka = ("mb", "ma") if use_mertens else ("bef", "aft")
        b = [r[kb] for r in cl if r[kb] is not None]
        a = [r[ka] for r in cl if r[ka] is not None]
        worse = sum(1 for x, y in zip(b, a) if abs(y) > abs(x))
        print(f"  {name:34s} n={len(b):4d}  before={st.median(b):+6.2f}% -> after={st.median(a):+6.2f}%  "
              f"|after|={st.median([abs(y) for y in a]):.2f}%  worse={worse}/{len(b)} ({100*worse/max(len(b),1):.1f}%)")
        return cl

    print("\n=== Prospective A/B Summary (Table 3 & Fig. 5) ===")
    cl1 = _summ("v1 legacy (Aug 17 - Nov 5, enh)", v1)
    cl2 = _summ("v2 post-Nov-5 (Nov 9-23, enh)", v2)
    cl3e = _summ("bayes3 live (Nov 24, enh)", b3, use_mertens=False)
    cl3m = _summ("bayes3 live (Nov 24, Mertens lin)", b3, use_mertens=True)
    if cl3m:
        print(f"  bayes3 per-channel Mertens BGR:   "
              f"[{st.median([r['ab'] for r in cl3m]):+.2f}%, "
              f"{st.median([r['ag'] for r in cl3m]):+.2f}%, "
              f"{st.median([r['ar'] for r in cl3m]):+.2f}%]")

    # Daily medians for plotting
    by_day = collections.defaultdict(lambda: [[], []])
    for r in cl1 + cl2:
        d = r["dt"].date()
        by_day[d][0].append(r["bef"])
        by_day[d][1].append(r["aft"])

    days = sorted(by_day)
    med_b = [st.median(by_day[d][0]) for d in days]
    med_a = [st.median(by_day[d][1]) for d in days]

    fig, ax = plt.subplots(figsize=(8.8, 3.6), dpi=200)
    ax.plot(days, med_b, "o-", ms=3.5, lw=1.2, color="#d62728", alpha=0.85, label="Uncorrected enhanced dip (daily median)")
    ax.plot(days, med_a, "s-", ms=3.5, lw=1.3, color="#1f77b4", alpha=0.9, label="Corrected enhanced dip (legacy v1/v2)")
    if cl3e:
        d3 = datetime.date(2026, 11, 24)
        ax.scatter([d3], [st.median([r["aft"] for r in cl3e])], marker="*", s=140, color="#2ca02c", zorder=6,
                   label=f"2026-11-24 bayes3 enhanced ({st.median([r['aft'] for r in cl3e]):+.2f}%)")
        ax.scatter([d3], [st.median([r["ma"] for r in cl3m])], marker="D", s=65, color="#111111", zorder=7,
                   label=f"2026-11-24 bayes3 Mertens linear ({st.median([r['ma'] for r in cl3m]):+.2f}%)")
    ax.axhline(0, color="gray", lw=0.9, ls="--")
    ax.axvline(datetime.date(2026, 11, 5), color="purple", lw=0.9, ls=":", alpha=0.7)
    ax.text(datetime.date(2026, 11, 5), 14.5, " Nov 5\n (v2)", color="purple", fontsize=7.8, va="top")
    ax.set_ylabel("Clean-sky median dip (%)")
    ax.set_title("Prospective Nightly A/B Verification (Aug 17 – Nov 24, 2026)")
    ax.set_ylim(-4, 16)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False, loc="upper left", fontsize=7.8, ncol=2)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    p = out_dir / "ab_longitudinal.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ Fig. 7: 125-day Andrade Creep
def make_creep_4month(out_dir):
    pts = []
    # v1 tracker (Jul 22 - Nov 5)
    with open(DATA / "history_v1.csv") as fh:
        for r in csv.DictReader(fh):
            if r.get("accepted") == "1":
                d = datetime.datetime.strptime(r["day"], "%Y%m%d")
                pts.append((d, float(r["cx"]), float(r["cy"]), "v1"))
    # v2/v3 tracker (Nov 5 - Nov 24)
    with open(DATA / "history.csv") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "ok":
                d = datetime.datetime.strptime(r["day"], "%Y%m%d")
                pts.append((d, float(r["cx"]), float(r["cy"]), "v2"))

    # Deduplicate by date (prefer latest)
    by_date = {}
    for p in pts:
        by_date[p[0]] = p
    pts = [by_date[k] for k in sorted(by_date)]

    t0 = pts[0][0]
    days = np.array([(p[0] - t0).days for p in pts], dtype=np.float64)
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    x0, y0 = xs[0], ys[0]

    # Principal motion axis
    dx, dy = xs - x0, ys - y0
    u = np.array([dx[-1], dy[-1]])
    u = u / np.linalg.norm(u)
    s_along = dx * u[0] + dy * u[1]
    s_perp = -dx * u[1] + dy * u[0]
    total_disp = float(np.hypot(dx[-1], dy[-1]))

    # Fit Andrade power law s(t) = A * (t + t_off)^alpha - c
    # Standard 2-parameter power law from t>=1: s(t) = A * t^alpha
    mpos = days >= 1
    def power_law(t, A, alpha):
        return A * np.power(t, alpha)
    popt, _ = curve_fit(power_law, days[mpos], s_along[mpos], p0=[60.0, 0.33], maxfev=10000)
    A_fit, alpha_fit = float(popt[0]), float(popt[1])
    pred = np.where(days > 0, power_law(np.maximum(days, 1e-6), A_fit, alpha_fit), 0.0)
    ss_res = float(np.sum((s_along - pred) ** 2))
    ss_tot = float(np.sum((s_along - np.mean(s_along)) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    # Also fit exponential relaxation for comparison
    def exp_law(t, S_inf, tau):
        return S_inf * (1.0 - np.exp(-t / tau))
    popt_e, _ = curve_fit(exp_law, days, s_along, p0=[320.0, 35.0], maxfev=10000)
    pred_e = exp_law(days, *popt_e)
    r2_e = 1.0 - float(np.sum((s_along - pred_e) ** 2)) / ss_tot

    print("\n=== 125-Day Viscoelastic Mount Creep (Fig. 7) ===")
    print(f"  Accepted nightly fixes: {len(pts)} over {int(days[-1])} days ({pts[0][0].date()} -> {pts[-1][0].date()})")
    print(f"  Start ({x0:.1f}, {y0:.1f}) -> End ({xs[-1]:.1f}, {ys[-1]:.1f}); total disp = {total_disp:.1f} px; "
          f"transverse RMS = {np.sqrt(np.mean(s_perp**2)):.1f} px")
    print(f"  Andrade power law s(t) = {A_fit:.1f} * t^{alpha_fit:.3f}: R^2 = {r2:.4f}")
    print(f"  Exponential relaxation (S_inf={popt_e[0]:.1f} px, tau={popt_e[1]:.1f} d): R^2 = {r2_e:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200)
    sc = ax1.scatter(xs, ys, c=days, cmap="viridis", s=26, zorder=3)
    ax1.plot([x0, xs[-1]], [y0, ys[-1]], "k--", lw=1.0, alpha=0.5, label=f"Straight chord ({total_disp:.0f} px)")
    ax1.axhline(20, color="crimson", ls=":", lw=1.0, label="Top mask ($y=20$)")
    ax1.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax1, shrink=0.88, pad=0.02)
    cb.set_label("Days since 2026-07-22")
    ax1.set_xlabel("Shadow centre $x$ (px)")
    ax1.set_ylabel("Shadow centre $y$ (px)")
    ax1.set_title(f"2D Shadow Trajectory ({len(pts)} fixes, $\\perp$ RMS = {np.sqrt(np.mean(s_perp**2)):.1f} px)")
    ax1.legend(frameon=False, loc="lower left", fontsize=7.8)
    ax1.grid(alpha=0.25)

    tgrid = np.linspace(0.5, days[-1], 200)
    ax2.scatter(days, s_along, s=22, color="#1f77b4", alpha=0.85, label="Measured displacement $s(t)$")
    ax2.plot(tgrid, power_law(tgrid, A_fit, alpha_fit), "r-", lw=2.0,
             label=f"Andrade $A t^{{\\alpha}}$ ($\\alpha={alpha_fit:.2f}$, $R^2={r2:.3f}$)")
    ax2.plot(tgrid, exp_law(tgrid, *popt_e), "k--", lw=1.3, alpha=0.75,
             label=f"Exponential ($\\tau={popt_e[1]:.0f}$ d, $R^2={r2_e:.3f}$)")
    ax2.set_xlabel("Elapsed time $t$ (days since 2026-07-22)")
    ax2.set_ylabel("Along-track displacement $s(t)$ (px)")
    ax2.set_title("Viscoelastic Mount Creep")
    ax2.legend(frameon=False, loc="lower right", fontsize=8.0)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    p = out_dir / "creep_4month.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Reproduce v12 paper figures and stats from ./data/.")
    ap.add_argument("--out", default=str(ROOT / "reproduced_figs"), help="Output directory for PDFs")
    ap.add_argument("--overwrite-paper-figs", action="store_true", help="Write directly into ./paper/figs/")
    ap.add_argument("--run-stan", action="store_true", help="Also refit spot_m2.stan with cmdstanpy")
    args = ap.parse_args()

    out_dir = ROOT / "paper/figs" if args.overwrite_paper_figs else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = [
        make_crf_and_m2(out_dir, run_stan=args.run_stan),
        make_profile_v12(out_dir),
        make_ab_longitudinal(out_dir),
        make_creep_4month(out_dir),
    ]
    print("\nRegenerated figures:")
    for f in figs:
        print(f"  {f} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
