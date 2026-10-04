"""Self-contained HTML "selection-adjusted tear sheet"."""

from __future__ import annotations

import datetime as dt
import html
import math
from pathlib import Path

from .analysis import Analysis
from .outofsample import OutOfSample
from .simulation import CoverageResult
from .winners import Interval

__all__ = ["render_report", "write_report"]

_CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --neutral:#f0efec; --surface:#fcfcfb; --blue:#2a78d6; --green:#1baf7a; --orange:#eb6834;
  --red:#c93a39; --amber:#9a6a00; }
* { box-sizing: border-box; }
body { margin:0; background:#fff; color:var(--ink);
  font:15px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif; }
.page { max-width:900px; margin:0 auto; padding:56px 40px 64px; }
h1 { font-size:26px; font-weight:600; letter-spacing:-0.01em; margin:0 0 4px; }
h2 { font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;
  color:var(--ink2); margin:44px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--grid); }
.sub { color:var(--ink2); margin:0; }
.meta { color:var(--muted); font-size:13px; margin-top:6px; }
.tiles { display:flex; gap:16px; margin-top:32px; }
.tile { flex:1; background:var(--surface); border:1px solid var(--grid); border-radius:10px;
  padding:16px 18px; }
.tile .k { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }
.tile .v { font-size:30px; font-weight:600; letter-spacing:-0.02em; margin-top:2px; }
.tile .s { font-size:13px; color:var(--ink2); margin-top:2px; }
table { width:100%; border-collapse:collapse; font-size:14px; }
th { text-align:left; font-weight:500; color:var(--muted); font-size:12px;
  border-bottom:1px solid var(--axis); padding:6px 10px 6px 0; }
td { padding:8px 10px 8px 0; border-bottom:1px solid var(--grid); }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; padding-right:18px; }
tr.hl td { font-weight:600; }
.note { color:var(--ink2); font-size:13px; }
.warn { border-left:3px solid var(--amber); background:#fbf6ea; padding:10px 14px; margin:10px 0;
  font-size:13.5px; color:#5c4300; }
svg text { font-family:"Helvetica Neue", Helvetica, Arial, sans-serif; }
ul { padding-left:20px; margin:8px 0; } li { margin:4px 0; }
#summary { padding:20px 20px 4px; }
.foot { margin-top:48px; font-size:12px; color:var(--muted); }
"""

_COLOURS = {
    "naive": "#eb6834",
    "conditional": "#1baf7a",
    "hybrid": "#2a78d6",
    "projection": "#898781",
    "bonferroni": "#c3c2b7",
}


def _f(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def _pct(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.0%}"


def _ivl(i: Interval, digits: int = 2) -> str:
    return f"[{_f(i.lower, digits)}, {_f(i.upper, digits)}]"


def _nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step) * step
    ticks = []
    t = start
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _interval_svg(a: Analysis) -> str:
    w = a.winner
    rows = [
        ("Naive", "naive", w.estimate, w.naive),
        ("Conditional", "conditional", w.conditional_estimate, w.conditional),
        ("Hybrid", "hybrid", w.hybrid_estimate, w.hybrid),
        ("Projection", "projection", w.estimate, w.projection),
        ("Bonferroni", "bonferroni", w.estimate, w.bonferroni),
    ]
    lo = min(r[3].lower for r in rows)
    hi = max(r[3].upper for r in rows)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    left, right, top, rh = 110, 24, 14, 38
    width = 760
    height = top + rh * len(rows) + 34

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (width - left - right)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Naive and corrected intervals for the winner">'
    ]
    for t in _nice_ticks(lo, hi):
        parts.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" '
            f'y2="{top + rh * len(rows)}" stroke="#e1e0d9" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x(t):.1f}" y="{top + rh * len(rows) + 18}" font-size="11" '
            f'fill="#898781" text-anchor="middle">{t:g}</text>'
        )
    if lo < 0 < hi:
        parts.append(
            f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top}" '
            f'y2="{top + rh * len(rows)}" stroke="#898781" stroke-width="1.2"/>'
        )
    for i, (label, key, est, iv) in enumerate(rows):
        yc = top + rh * i + rh / 2
        c = _COLOURS[key]
        parts.append(
            f'<text x="{left - 12}" y="{yc + 4:.1f}" font-size="12.5" fill="#52514e" '
            f'text-anchor="end">{label}</text>'
        )
        parts.append(
            f'<line x1="{x(iv.lower):.1f}" x2="{x(iv.upper):.1f}" y1="{yc:.1f}" '
            f'y2="{yc:.1f}" stroke="{c}" stroke-width="5" stroke-linecap="round"/>'
        )
        parts.append(
            f'<circle cx="{x(est):.1f}" cy="{yc:.1f}" r="5.5" fill="#fff" '
            f'stroke="{c}" stroke-width="2.5"/>'
        )
    unit = "annualised Sharpe ratio" if a.performance.statistic == "sharpe" else "annualised mean"
    parts.append(
        f'<text x="{(left + width - right) / 2:.1f}" y="{height - 2}" font-size="11.5" '
        f'fill="#52514e" text-anchor="middle">{unit}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _coverage_svg(c: CoverageResult, alpha: float) -> str:
    rows = [
        ("Naive", "naive", c.naive_coverage),
        ("Conditional", "conditional", c.conditional_coverage),
        ("Hybrid", "hybrid", c.hybrid_coverage),
        ("Projection", "projection", c.projection_coverage),
        ("Bonferroni", "bonferroni", c.bonferroni_coverage),
    ]
    left, right, top, rh = 110, 56, 10, 34
    width = 760
    height = top + rh * len(rows) + 34
    lo, hi = 0.0, 1.0

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (width - left - right)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Simulated coverage of each interval">'
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" '
            f'y2="{top + rh * len(rows)}" stroke="#e1e0d9"/>'
        )
        parts.append(
            f'<text x="{x(t):.1f}" y="{top + rh * len(rows) + 18}" font-size="11" '
            f'fill="#898781" text-anchor="middle">{t:.0%}</text>'
        )
    nominal = 1.0 - alpha
    parts.append(
        f'<line x1="{x(nominal):.1f}" x2="{x(nominal):.1f}" y1="{top - 4}" '
        f'y2="{top + rh * len(rows)}" stroke="#0b0b0b" stroke-width="1.2" '
        f'stroke-dasharray="4 3"/>'
    )
    for i, (label, key, cov) in enumerate(rows):
        yc = top + rh * i + rh / 2
        parts.append(
            f'<text x="{left - 12}" y="{yc + 4:.1f}" font-size="12.5" fill="#52514e" '
            f'text-anchor="end">{label}</text>'
        )
        parts.append(
            f'<rect x="{x(0):.1f}" y="{yc - 8:.1f}" width="{x(cov) - x(0):.1f}" '
            f'height="16" rx="3" fill="{_COLOURS[key]}"/>'
        )
        parts.append(
            f'<text x="{x(cov) + 8:.1f}" y="{yc + 4:.1f}" font-size="12" fill="#0b0b0b" '
            f'font-variant-numeric="tabular-nums">{cov:.1%}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def render_report(
    a: Analysis,
    coverage: CoverageResult | None = None,
    *,
    source: str | None = None,
    title: str = "Selection-adjusted tear sheet",
    oos: OutOfSample | None = None,
) -> str:
    """Render ``a`` (and an optional coverage simulation) as a standalone HTML page."""
    w, p, d = a.winner, a.performance, a.diagnostics
    stat = "Sharpe ratio" if p.statistic == "sharpe" else "mean return"
    level = f"{100 * (1 - w.alpha):g}%"
    head_iv = a.headline_interval
    meta_bits = [f"{p.n_strategies} trials", f"{p.n_obs} observations"]
    if a.period:
        meta_bits.append(f"{a.period[0]} to {a.period[1]}")
    meta_bits.append(f"{p.periods_per_year:g} periods per year")
    cov_label = {
        "hac": "HAC (Newey-West)",
        "iid": "iid",
        "normal": "iid normal (Lo/Memmel)",
        "bootstrap": "stationary bootstrap",
    }[p.cov_method]
    meta_bits.append(
        f"covariance: {cov_label}" + (" + shrinkage" if p.shrinkage is not None else "")
    )
    src = f"{html.escape(source)} &middot; " if source else ""
    meta = " &middot; ".join(html.escape(b) for b in meta_bits)

    rows = [
        ("Naive", w.estimate, w.naive, "", False),
        ("Conditional", w.conditional_estimate, w.conditional, "median-unbiased", False),
        ("Hybrid", w.hybrid_estimate, w.hybrid, "default; bounded by projection", True),
        ("Projection", float("nan"), w.projection, "simultaneous; no point estimate", False),
        ("Bonferroni", float("nan"), w.bonferroni, "ignores correlation", False),
    ]
    table = "\n".join(
        f'<tr{" class=hl" if hl else ""}><td>{name}</td><td class="n">{_f(est)}</td>'
        f'<td class="n">{_ivl(iv)}</td><td class="n">{_f(iv.width)}</td>'
        f'<td class="note">{html.escape(note)}</td></tr>'
        for name, est, iv, note, hl in rows
    )

    comp = ""
    if a.dsr is not None and a.haircuts is not None:
        s = a.dsr
        hb, hy = a.haircuts["bonferroni"], a.haircuts["bhy"]
        arch_rows = ""
        if a.arch is not None:
            sup = "yes" if a.arch.winner_is_superior else "no"
            arch_rows = (
                f'<tr><td>Hansen SPA</td><td class="n">p = {a.arch.spa_pvalue:.3f}</td>'
                '<td class="note">null: no strategy has a positive mean return</td></tr>'
                f'<tr><td>Romano-Wolf StepM</td><td class="n">{sup}</td>'
                f'<td class="note">winner has a positive mean at size {a.arch.size:g} '
                f"({len(a.arch.superior)} strategies identified)</td></tr>"
            )
        comp = f"""
<h2>Comparators</h2>
<p class="note">These tests ask whether the winner differs from zero. None of them gives a
corrected estimate or an interval for its true {stat}.</p>
<table>
<tr><th>Method</th><th class="n">Result</th><th>Reading</th></tr>
<tr><td>Deflated Sharpe ratio, N = K = {s.n_trials:.0f}</td><td class="n">{_f(s.dsr, 3)}</td>
<td class="note">probability that the true {stat} exceeds the expected maximum of
{s.n_trials:.0f} unskilled trials (annualised {_f(s.expected_max_sharpe * math.sqrt(p.periods_per_year))})</td></tr>
<tr><td>Deflated Sharpe ratio, N = {s.effective_trials:.1f} effective</td>
<td class="n">{_f(s.dsr_effective, 3)}</td>
<td class="note">same with the eigenvalue-based effective number of trials</td></tr>
<tr><td>Harvey-Liu haircut, Bonferroni (= Holm)</td><td class="n">{hb.haircut:.0%}</td>
<td class="note">adjusted p = {_f(hb.adjusted_p_value, 3)}, adjusted {stat} {_f(hb.adjusted_sharpe)}</td></tr>
<tr><td>Harvey-Liu haircut, BHY</td><td class="n">{hy.haircut:.0%}</td>
<td class="note">adjusted p = {_f(hy.adjusted_p_value, 3)}, adjusted {stat} {_f(hy.adjusted_sharpe)}</td></tr>
{arch_rows}</table>"""

    lo_b, hi_b = w.truncation
    bound_txt = (
        f"[{_f(lo_b)}, {'+inf' if not math.isfinite(hi_b) else _f(hi_b)}]"
        if math.isfinite(lo_b)
        else f"(-inf, {_f(hi_b)}]"
    )
    diag = f"""
<h2>Diagnostics</h2>
<table>
<tr><td>Winner margin</td><td class="n">{_f(d.margin)} SE</td>
<td class="note">smallest studentised gap to a rival (trial {html.escape(p.names[d.runner_up])})</td></tr>
<tr><td>Gap to second best</td><td class="n">{_f(d.gap_in_se)} SE</td>
<td class="note">difference of the two largest estimates in standard errors of the winner</td></tr>
<tr><td>Effective number of trials</td><td class="n">{_f(d.effective_trials, 1)}</td>
<td class="note">of {d.n_trials} logged; eigenvalue participation ratio of the correlations</td></tr>
<tr><td>Truncation interval for the winner</td><td class="n">{bound_txt}</td>
<td class="note">selection holds iff the winner's estimate lies here given the rest of the data</td></tr>
<tr><td>Conditional / naive width</td><td class="n">{_f(d.conditional_to_naive_width, 1)}x</td>
<td class="note">large when the winner barely wins; the hybrid interval caps it</td></tr>
</table>"""
    warns = "".join(f'<div class="warn">{html.escape(m)}</div>' for m in a.warnings)

    cov_block = ""
    if coverage is not None:
        cov_block = f"""
<h2>Coverage under your covariance</h2>
<p class="note">{coverage.n_rep:,} simulated panels with the estimated covariance matrix in which
every trial has the same true {stat} (the least-separated configuration), the best trial is
selected and each interval is computed. The dashed line is the nominal {level}. Failures of the
inversion: {coverage.n_failed} (counted as misses). Simulation error is about
&plusmn;{1.96 * math.sqrt(0.95 * 0.05 / coverage.n_rep):.1%}.</p>
{_coverage_svg(coverage, w.alpha)}"""

    sens = ""
    if a.projection_sensitivity:
        ps = a.projection_sensitivity
        lines = "".join(
            f"<tr><td>{r.n_hidden} hidden trials</td><td class='n'>{_f(r.critical_value)}</td>"
            f"<td class='n'>[{_f(r.lower)}, {_f(r.upper)}]</td><td class='n'>{_f(r.width_ratio)}x</td></tr>"
            for r in ps
        )
        sc = a.scenario
        scen = ""
        if sc is not None:
            scen = (
                f"<p class='note'>Scenario simulation (not a confidence interval): if every trial, "
                f"logged or hidden, had true {stat} {sc.true_value:g} and each hidden trial had "
                f"correlation {sc.correlation:.2f} with the logged trial it resembles, the winner "
                f"would still be the best "
                f"in {sc.acceptance:.0%} of draws. Over those draws the hybrid estimate ranges from "
                f"{_f(sc.hybrid_estimate[0])} to {_f(sc.hybrid_estimate[2])} (median "
                f"{_f(sc.hybrid_estimate[1])}) and the hybrid interval's lower end from "
                f"{_f(sc.hybrid_lower[0])} to {_f(sc.hybrid_lower[2])}.</p>"
            )
        sens = f"""
<h2>Unlogged trials</h2>
<p class="note">Exact for the projection interval, which is valid for any selection among the
enlarged set; the conditional and hybrid intervals depend on unobserved values and are shown
only as a scenario.</p>
<table><tr><th>Assumption</th><th class="n">Critical value</th><th class="n">Projection interval</th>
<th class="n">Width vs logged</th></tr>{lines}</table>{scen}"""

    oos_block = ""
    if oos is not None:
        fc = oos.forecast
        mark = {True: "inside", False: "outside"}
        oos_rows = "".join(
            f"<tr><td>{name}</td><td class='n'>{_ivl(iv)}</td><td class='n'>{mark[inside]}</td></tr>"
            for name, iv, inside in (
                ("Naive", fc.naive, oos.in_naive),
                ("Conditional", fc.conditional, oos.in_conditional),
                ("Hybrid", fc.hybrid, oos.in_hybrid),
            )
        )
        oos_block = f"""
<h2>Out-of-sample check</h2>
<p class="note">Realised {stat} of the winner on {oos.n_obs} later observations:
<b>{_f(oos.realised)}</b> (standard error {_f(oos.se)}). The intervals below forecast this new
estimate if the winner's true {stat} were unchanged; they include its own sampling error, which
an interval for the in-sample true value does not. Conditional p-value: {oos.conditional_pvalue:.3f}.
A realised value outside the interval points to decay beyond what selection explains.</p>
<table><tr><th>Forecast interval</th><th class="n">{level} interval</th>
<th class="n">Realised value</th></tr>{oos_rows}</table>"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body><div class="page">
<div id="summary">
<h1>{html.escape(title)}</h1>
<p class="sub">The best of {p.n_strategies} backtests, corrected for the selection that picked it.</p>
<p class="meta">{src}{meta}</p>

<div class="tiles">
<div class="tile"><div class="k">Winner</div><div class="v" style="font-size:20px;line-height:1.3">{html.escape(a.winner_name)}</div>
<div class="s">trial {w.index + 1} of {p.n_strategies}</div></div>
<div class="tile"><div class="k">Naive {stat}</div><div class="v">{_f(w.estimate)}</div>
<div class="s">{level} interval {_ivl(w.naive)}</div></div>
<div class="tile"><div class="k">Corrected ({a.method})</div><div class="v">{_f(a.headline_estimate)}</div>
<div class="s">{level} interval {_ivl(head_iv)}</div></div>
<div class="tile"><div class="k">Haircut</div><div class="v">{_pct(a.haircut)}</div>
<div class="s">of the naive {stat}</div></div>
</div>
{warns}
<h2>Estimates and intervals for the winner's true {stat}</h2>
<table><tr><th>Procedure</th><th class="n">Estimate</th><th class="n">{level} interval</th>
<th class="n">Width</th><th>Note</th></tr>{table}</table>
<div style="margin-top:18px">{_interval_svg(a)}</div>
<p class="note">Circles mark point estimates (naive: the winner's observed value). The conditional
estimate is median-unbiased given selection; the hybrid estimate is within &beta;/2 of
median-unbiased.</p>
</div>
{comp}{diag}{cov_block}{sens}{oos_block}
<p class="foot">Method: conditional and hybrid inference on winners (Andrews, Kitagawa and McCloskey,
Quarterly Journal of Economics 139(1), 2024). Under normality with a known covariance the
conditional interval has exact conditional coverage and the hybrid interval has unconditional
coverage of at least the nominal level; with an estimated covariance and non-normal returns both
hold asymptotically. Selection is argmax on the same data; discretionary choices, hidden trials
and non-stationarity are not covered.
Seed {a.seed}, {a.n_sim:,} critical-value draws, generated {dt.date.today().isoformat()}.
Illustrative research, not investment advice.</p>
</div></body></html>
"""


def write_report(
    path: str | Path, a: Analysis, coverage: CoverageResult | None = None, **kwargs
) -> Path:
    """Write the tear sheet to ``path`` and return it."""
    path = Path(path)
    path.write_text(render_report(a, coverage, **kwargs), encoding="utf-8")
    return path
