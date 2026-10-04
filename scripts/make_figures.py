"""Draw every README figure from ``docs/results.json`` (light and dark variants).

Colours are fixed per procedure in every figure: hybrid blue, naive orange, conditional
green, projection amber. Data are computed (or read) first; the ``build`` functions only
draw.
"""

from __future__ import annotations

import warnings

import numpy as np
from common import FIGURES, load_results
from matplotlib.lines import Line2D

from wincurse import WincurseWarning, independent_critical_value, winner_inference
from wincurse.viz import save_figure

KEYS = ("naive", "conditional", "hybrid", "projection")
LABELS = {
    "naive": "Naive",
    "conditional": "Conditional",
    "hybrid": "Hybrid",
    "projection": "Projection",
}


def colours(p) -> dict[str, str]:
    s = p["series"]
    return {"hybrid": s[0], "naive": s[1], "conditional": s[2], "projection": s[3]}


# ---------------------------------------------------------------------------- figure 1


def coverage_vs_k(results: dict) -> None:
    known = results["coverage"]["known_sigma"]
    feasible = results["coverage"]["feasible"]
    reps = results["coverage"]["design"]["reps_known_sigma"]
    rhos = (0.3, 0.5, 0.7, 0.9)
    ks = (10, 100, 1000)
    curves = {
        (rho, key): [known[f"known_K{k}_rho{rho}_equal_n{reps}"][f"{key}_coverage"] for k in ks]
        for rho in rhos
        for key in ("naive", "conditional", "hybrid")
    }
    n_feas = results["coverage"]["design"]["reps_feasible"]
    points = {}
    for rho in rhos:
        for k in ks:
            cell = feasible.get(f"feas_K{k}_rho{rho}_equal_normal_hac_n{n_feas}")
            if cell:
                points.setdefault(rho, []).append((k, cell["estimated_sigma"]["hybrid_coverage"]))

    def build(p):
        import matplotlib.pyplot as plt

        c = colours(p)
        fig, axes = plt.subplots(1, 4, figsize=(11, 3.5), sharey=True)
        for ax, rho in zip(axes, rhos, strict=True):
            ax.axhline(0.95, color=p["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.plot(ks, curves[(rho, "naive")], marker="o", color=c["naive"], zorder=3)
            ax.plot(ks, curves[(rho, "hybrid")], marker="o", color=c["hybrid"], lw=3.4, zorder=3)
            ax.plot(
                ks,
                curves[(rho, "conditional")],
                marker="s",
                ms=4.5,
                lw=1.4,
                color=c["conditional"],
                zorder=4,
            )
            if rho in points:
                kx, ky = zip(*points[rho], strict=True)
                ax.scatter(
                    kx,
                    ky,
                    s=46,
                    marker="D",
                    facecolors=p["surface"],
                    edgecolors=c["hybrid"],
                    linewidths=1.6,
                    zorder=4,
                )
            ax.set_xscale("log")
            ax.set_xticks(ks, [str(k) for k in ks])
            ax.minorticks_off()
            ax.set_ylim(0, 1.02)
            ax.set_title(f"correlation {rho:g}", fontsize=10)
            ax.set_xlabel("number of trials K")
        axes[0].set_ylabel("coverage of the winner's true Sharpe ratio")
        axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        handles = [
            Line2D([], [], color=c["naive"], marker="o", label="Naive"),
            Line2D([], [], color=c["conditional"], marker="s", ms=4.5, label="Conditional"),
            Line2D([], [], color=c["hybrid"], marker="o", label="Hybrid (known covariance)"),
            Line2D(
                [],
                [],
                color=c["hybrid"],
                marker="D",
                ls="",
                markerfacecolor=p["surface"],
                label="Hybrid (covariance estimated)",
            ),
            Line2D([], [], color=p["muted"], ls=(0, (4, 3)), label="Nominal 95%"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
        fig.suptitle(
            "Best of K backtests: the naive 95% interval covers far less, corrected ones about 95%",
            x=0.01,
            ha="left",
            fontsize=12,
            fontweight="bold",
            y=1.03,
        )
        fig.tight_layout()
        return fig

    save_figure(build, FIGURES / "coverage_vs_k")


# ---------------------------------------------------------------------------- figure 2


def real_data(results: dict) -> None:
    panels = []
    for key, title in (
        ("momentum", "Industry momentum, 256 variants"),
        ("technical", "Technical trading rules, 340 variants"),
    ):
        r = results[key]
        w, o = r["winner"], r["out_of_sample_check"]
        rows = [
            ("Naive", w["naive_interval"], w["naive"], "naive"),
            ("Conditional", w["conditional_interval"], w["conditional_estimate"], "conditional"),
            ("Hybrid", w["hybrid_interval"], w["hybrid_estimate"], "hybrid"),
            ("Projection", w["projection_interval"], None, "projection"),
            None,
            ("Naive forecast", o["forecast_naive"], None, "naive"),
            ("Hybrid forecast", o["forecast_hybrid"], None, "hybrid"),
        ]
        sub = (
            f"{r['in_sample']['start'][:4]}-{r['in_sample']['end'][:4]} in sample, "
            f"{r['out_of_sample']['start'][:4]}-{r['out_of_sample']['end'][:4]} out of sample"
        )
        panels.append((title, sub, rows, o["realised"], w["name"]))

    def build(p):
        import matplotlib.pyplot as plt

        c = colours(p)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
        for ax, (title, sub, rows, realised, _name) in zip(axes, panels, strict=True):
            y = 0
            ticks, labels = [], []
            for row in rows:
                if row is None:
                    y += 1
                    continue
                label, (lo, hi), est, key = row
                ax.plot([lo, hi], [-y, -y], color=c[key], lw=5, solid_capstyle="round", zorder=3)
                if est is not None:
                    ax.scatter(
                        [est],
                        [-y],
                        s=60,
                        facecolors=p["surface"],
                        edgecolors=c[key],
                        linewidths=2.2,
                        zorder=4,
                    )
                ticks.append(-y)
                labels.append(label)
                y += 1
            ax.axvline(realised, color=p["ink"], lw=1.4, ls=(0, (5, 3)), zorder=2)
            ax.text(
                realised,
                -4.0,
                f"  realised out of sample: {realised:.2f}",
                ha="left",
                va="center",
                fontsize=8.5,
                color=p["ink"],
            )
            ax.axvline(0, color=p["axis"], lw=1.0, zorder=1)
            ax.set_yticks(ticks, labels)
            ax.grid(axis="y", visible=False)
            ax.set_ylim(-y + 0.4, 0.8)
            ax.set_xlabel("annualised Sharpe ratio")
            ax.set_title(title, fontsize=10.5, pad=20)
            ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=8.5, color=p["muted"])
        fig.suptitle(
            "Technical rules decay out of sample far beyond selection; industry momentum holds up",
            x=0.01,
            ha="left",
            fontsize=12,
            fontweight="bold",
            y=1.04,
        )
        fig.tight_layout()
        return fig

    save_figure(build, FIGURES / "real_data")


# ---------------------------------------------------------------------------- figure 3


def endpoints_vs_gap() -> None:
    gaps = np.logspace(-3, np.log10(6.0), 70)
    sigma = np.eye(2)
    crit = (independent_critical_value(2, 0.05), independent_critical_value(2, 0.005))
    lower = {key: [] for key in KEYS}
    upper = {key: [] for key in KEYS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WincurseWarning)
        for g in gaps:
            res = winner_inference(np.array([g, 0.0]), sigma, 0.05, 0.005, critical_values=crit)
            for key in KEYS:
                lower[key].append(getattr(res, key).lower - g)
                upper[key].append(getattr(res, key).upper - g)
    lead = gaps / np.sqrt(2.0)

    def build(p):
        import matplotlib.pyplot as plt

        c = colours(p)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharex=True, sharey=True)
        for ax, data, name in ((axes[0], lower, "lower"), (axes[1], upper, "upper")):
            for key in KEYS:
                style = (0, (4, 3)) if key == "projection" else "-"
                ax.plot(lead, data[key], color=c[key], ls=style, label=LABELS[key])
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=5.0)
            ax.set_title(f"{name} end", fontsize=10.5)
            ax.set_xlabel("lead over the runner-up (standard errors of the difference)")
        axes[0].set_ylabel("interval end minus the winner's estimate (SE)")
        ticks = [-1000, -100, -10, -2, 0, 2, 10]
        axes[0].set_yticks(ticks, [str(t) for t in ticks])
        axes[1].legend(loc="center right")
        fig.suptitle(
            "Two independent trials: a narrow win sends the conditional interval to thousands of "
            "SE; the hybrid one stays inside the projection set",
            x=0.01,
            ha="left",
            fontsize=11.5,
            fontweight="bold",
            y=1.03,
        )
        fig.tight_layout()
        return fig

    save_figure(build, FIGURES / "endpoints_vs_gap")


# ---------------------------------------------------------------------------- figure 4


def robustness(results: dict) -> None:
    cells = results["coverage"]["feasible"]
    n_feas = results["coverage"]["design"]["reps_feasible"]
    dist_label = {
        "normal": "normal",
        "t5": "Student-t(5)",
        "garch": "GARCH(1,1)",
        "ar1": "AR(1) 0.2",
    }
    cov_label = {"normal": "Lo/Memmel", "iid": "iid robust", "hac": "HAC"}
    rows = []
    for dist in dist_label:
        for cov in cov_label:
            cell = cells.get(f"feas_K100_rho0.5_equal_{dist}_{cov}_n{n_feas}")
            if cell:
                rows.append((f"{dist_label[dist]}, {cov_label[cov]}", cell["estimated_sigma"]))
    for suffix, label in (
        ("_T504", "normal, HAC, 2 years daily"),
        ("_T120", "normal, HAC, 10 years monthly"),
    ):
        cell = cells.get(f"feas_K100_rho0.5_equal_normal_hac{suffix}_n{n_feas}")
        if cell:
            rows.append((label, cell["estimated_sigma"]))

    def build(p):
        import matplotlib.pyplot as plt

        c = colours(p)
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ys = -np.arange(len(rows))
        ax.axvline(0.95, color=p["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        for key in ("naive", "conditional", "hybrid"):
            ax.scatter([r[1][f"{key}_coverage"] for r in rows], ys, s=42, color=c[key], zorder=3)
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.45, 1.0)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("coverage of the winner's true Sharpe ratio (K = 100, correlation 0.5)")
        handles = [
            Line2D([], [], color=c[k], marker="o", ls="", label=LABELS[k])
            for k in ("naive", "conditional", "hybrid")
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
        ax.set_title("Returns and covariance estimator", fontsize=10.5)
        fig.tight_layout()
        return fig

    save_figure(build, FIGURES / "robustness")


def main() -> None:
    results = load_results()
    if "coverage" in results:
        coverage_vs_k(results)
        robustness(results)
    if "momentum" in results and "technical" in results:
        real_data(results)
    endpoints_vs_gap()
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
