"""Self-contained HTML dashboard rendered from saved records only.

No CDN, no JavaScript dependencies, no training code paths, no test data. Every
number shown comes from record.json, curves.jsonl, decisions.jsonl, the registry
index, the frozen manifest or final_eval/results.json. Missing pieces are shown as
missing; synthetic fixtures and screening runs are labelled.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

from gotrain.agent.session import list_sessions, session_dir
from gotrain.artifacts.store import FilesystemArtifactStore
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.report.png import colour_mask, png_data_uri
from gotrain.util.io import iter_jsonl, read_json
from gotrain.util.timing import utc_now_iso

REQUIRED_TASKS = ("m-eurosat", "m-pv4ger", "m-pv4ger-seg", "m-chesapeake")

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1d2330}
header{background:#14213d;color:#fff;padding:18px 28px}header h1{margin:0;font-size:22px}header .sub{opacity:.8;font-size:13px;margin-top:4px}
main{padding:20px 28px;max-width:1400px;margin:0 auto}
section{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:16px 20px;margin-bottom:18px}
h2{font-size:18px;margin:0 0 10px}h3{font-size:15px;margin:16px 0 6px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #eceff3;padding:5px 8px;text-align:left;vertical-align:top}th{background:#f1f3f6;font-weight:600}
tr.best td{background:#eefaf0}tr.screen td{color:#6b7280;font-style:italic}tr.failed td{background:#fff4f2}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600;margin-left:4px}
.b-warn{background:#fde68a;color:#78350f}.b-ok{background:#bbf7d0;color:#14532d}.b-bad{background:#fecaca;color:#7f1d1d}.b-info{background:#dbeafe;color:#1e3a8a}.b-grey{background:#e5e7eb;color:#374151}
.banner{background:#fde68a;border:2px solid #d97706;padding:10px 14px;border-radius:6px;font-weight:600;margin-bottom:16px}
.grid{display:flex;flex-wrap:wrap;gap:14px}.card{border:1px solid #e3e6ea;border-radius:6px;padding:10px 12px;min-width:220px;flex:1}
.card .big{font-size:24px;font-weight:700}.muted{color:#6b7280;font-size:12px}
.examples{display:flex;flex-wrap:wrap;gap:10px}.ex{text-align:center;font-size:11px}.ex img{image-rendering:pixelated;border:1px solid #d1d5db;background:#000}
.seg{display:inline-flex;gap:4px;border:1px solid #e3e6ea;padding:4px;border-radius:4px}
pre{background:#0f172a;color:#e2e8f0;padding:10px;border-radius:6px;font-size:12px;overflow-x:auto}
details{margin:6px 0}summary{cursor:pointer;font-weight:600}
.decision{border-left:3px solid #94a3b8;padding:4px 10px;margin:4px 0;font-size:13px}.d-propose{border-color:#2563eb}.d-result{border-color:#16a34a}.d-block{border-color:#dc2626}.d-select{border-color:#7c3aed}
svg text{font-size:10px;fill:#374151}
"""


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "<span class='muted'>n/a</span>"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return esc(x)


def badge(text: str, kind: str = "info") -> str:
    return f"<span class='badge b-{kind}'>{esc(text)}</span>"


def line_chart(series: list[dict], width: int = 520, height: int = 220, y_label: str = "", y_range: tuple[float, float] | None = None) -> str:
    """series: [{name, points: [(x, y)], color}]"""
    pts_all = [p for s in series for p in s["points"] if p[1] is not None]
    if not pts_all:
        return "<div class='muted'>no curve data</div>"
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    x0, x1 = min(xs), max(xs)
    y0, y1 = (y_range if y_range else (min(ys), max(ys)))
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1e-6
    ml, mr, mt, mb = 44, 12, 12, 28
    pw, ph = width - ml - mr, height - mt - mb

    def sx(x):
        return ml + (x - x0) / (x1 - x0) * pw

    def sy(y):
        return mt + (1 - (y - y0) / (y1 - y0)) * ph

    out = [f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"]
    out.append(f"<rect x='{ml}' y='{mt}' width='{pw}' height='{ph}' fill='#fff' stroke='#d1d5db'/>")
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        out.append(f"<line x1='{ml}' x2='{ml + pw}' y1='{sy(y):.1f}' y2='{sy(y):.1f}' stroke='#f1f3f6'/>")
        out.append(f"<text x='{ml - 4}' y='{sy(y) + 3:.1f}' text-anchor='end'>{y:.3g}</text>")
    for i in range(min(6, int(x1 - x0) + 1)):
        x = x0 + (x1 - x0) * i / max(1, min(5, int(x1 - x0)))
        out.append(f"<text x='{sx(x):.1f}' y='{height - 10}' text-anchor='middle'>{x:.0f}</text>")
    out.append(f"<text x='{ml + pw / 2:.0f}' y='{height - 1}' text-anchor='middle'>epoch</text>")
    if y_label:
        out.append(f"<text x='{ml + 4}' y='{mt + 10}' font-weight='bold'>{esc(y_label)}</text>")
    legend = []
    for s in series:
        pts = [(sx(x), sy(y)) for x, y in s["points"] if y is not None]
        if not pts:
            continue
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f"<polyline points='{d}' fill='none' stroke='{s['color']}' stroke-width='1.8'/>")
        legend.append(f"<span class='badge' style='background:{s['color']};color:#fff'>{esc(s['name'][:40])}</span>")
    out.append("</svg>")
    return "".join(out) + "<div class='muted'>" + " ".join(legend) + "</div>"


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]


def dashboard_tasks(records: list[dict], fixture: bool = False) -> list[str]:
    """Optional-only run homes have their own scope; empty/main homes retain required-task warnings."""
    actual = {r["task"] for r in records}
    required = set(REQUIRED_TASKS)
    expected = required if not fixture and (not actual or actual & required) else set()
    return sorted(actual | expected)


class DashboardData:
    def __init__(self, home: Home) -> None:
        self.home = home
        self.store = ExperimentStore(home)
        self.registry = FilesystemArtifactStore(home)
        self.records = self.store.all_records()
        self.fixture = (home.root / "FIXTURE.json").exists()
        self.sessions = list_sessions(home)
        self.manifest = read_json(home.selection / "manifest.json") if (home.selection / "manifest.json").exists() else None
        self.final = read_json(home.final_eval / "results.json") if (home.final_eval / "results.json").exists() else None
        self.tasks = dashboard_tasks(self.records, self.fixture)

    def curves(self, exp_id: str) -> list[dict]:
        return list(iter_jsonl(self.store.exp_dir(exp_id) / "checkpoints" / "curves.jsonl"))

    def examples(self, rec: dict) -> dict | None:
        info = rec.get("val_examples")
        if not info:
            return None
        path = self.store.exp_dir(rec["experiment_id"]) / info["path"]
        if not path.exists():
            return None
        try:
            return dict(np.load(path))
        except Exception:
            return None


def render_overview(d: DashboardData) -> str:
    rows = []
    for task in d.tasks:
        recs = [r for r in d.records if r["task"] == task]
        full = [r for r in recs if r["status"] == "completed" and not r.get("screening") and not r.get("partial_validation")]
        screen = [r for r in recs if r.get("screening") and r["status"] == "completed"]
        failed = [r for r in recs if r["status"] in ("failed", "interrupted")]
        best = d.store.best(task)
        base = [r for r in full if r.get("tag") == "baseline"]
        base_score = base[-1]["result"]["best_val_metric"] if base else None
        best_score = best["best_val_metric"] if best else None
        delta = (best_score - base_score) if (best_score is not None and base_score is not None) else None
        metric = recs[0]["metric"]["name"] if recs else ("accuracy" if task in ("m-eurosat", "m-pv4ger") else "miou")
        promoted = d.registry.promoted(task)
        final = (d.final or {}).get("tasks", {}).get(task)
        if final and final.get("status") == "completed":
            final_html = f"{final['test_metric']:.4f} {badge('test, once', 'ok')}"
        elif final:
            final_html = badge(f"final eval {final.get('status')}", "bad")
        else:
            final_html = badge("not evaluated", "grey")
        status = badge("no experiments", "warn") if not recs else (badge(f"{len(full)} full / {len(screen)} screening / {len(failed)} failed", "info"))
        rows.append(
            f"<tr><td><b>{esc(task)}</b></td><td>{esc(metric)}</td><td>{status}</td>"
            f"<td>{fmt(base_score)}</td><td>{fmt(best_score)} {('<span class=\"muted\">' + esc(best['experiment_id']) + ' by ' + esc(best['selected_by']) + '</span>') if best else ''}</td>"
            f"<td>{(('+' if delta >= 0 else '') + f'{delta:.4f}') if delta is not None else fmt(None)}</td>"
            f"<td>{esc(promoted['version_id']) if promoted else fmt(None)}</td><td>{final_html}</td></tr>"
        )
    return (
        "<section><h2>Task overview</h2><table><tr><th>task</th><th>frozen metric (validation)</th><th>experiments</th><th>fixed baseline</th><th>best full run (validation)</th><th>agent minus baseline</th><th>promoted version</th><th>final test score</th></tr>"
        + "".join(rows)
        + "</table><div class='muted'>Validation scores drive every ranking. Different tasks use different metrics and are never compared with each other. Test scores appear only after the one-shot final evaluation.</div></section>"
    )


def render_task(d: DashboardData, task: str) -> str:
    recs = sorted([r for r in d.records if r["task"] == task], key=lambda r: r["experiment_id"])
    if not recs:
        return f"<section><h2>{esc(task)}</h2><div class='banner'>No experiments recorded for this task yet.</div></section>"
    metric = recs[0]["metric"]["name"]
    best = d.store.best(task)
    best_id = best["experiment_id"] if best else None
    ranked = d.store.ranking(task, source=recs[0]["source"])
    others = [r for r in recs if r["experiment_id"] not in {x["experiment_id"] for x in ranked}]
    parts = [f"<section><h2>{esc(task)} <span class='muted'>{esc(recs[0]['task_type'])}, metric {esc(metric)} (max)</span></h2>"]
    parts.append(f"<div class='muted'>{esc(recs[0]['metric']['definition'])}</div>")

    def row(r: dict, cls: str = "") -> str:
        s = d.store.compact_summary(r)
        tags = badge(s["tag"], "info")
        if s["screening"]:
            tags += badge("screening, partial data", "warn")
        if r.get("partial_validation"):
            tags += badge("partial validation", "warn")
        if r.get("source") == "synthetic":
            tags += badge("synthetic", "bad")
        if s["adjustments"]:
            tags += badge(f"{len(s['adjustments'])} auto-adjustment(s)", "warn")
        klass = cls or ("best" if r["experiment_id"] == best_id else "")
        return (
            f"<tr class='{klass}'><td>{esc(s['experiment_id'])}<br>{tags}</td><td>{esc(s['status'])}<br><span class='muted'>{esc(s['stop_reason'] or s['error'] or '')}</span></td>"
            f"<td>{esc(s['model'])}</td><td>{fmt(s['lr'], 5)}</td><td>{esc(s['batch_size'])}</td><td>{esc(s['epochs_completed'])}/{esc(s['epochs_requested'])}</td>"
            f"<td>{esc(s['subset_fraction'])}</td><td>{esc(s['input_size'] or 'native')}</td><td><b>{fmt(s['best_val_metric'])}</b> @{esc(s['best_epoch'])}</td><td>{fmt(s['wall_minutes'], 1)}</td>"
            f"<td class='muted'>{esc((r.get('hypothesis') or '')[:140])}</td></tr>"
        )

    header = "<tr><th>experiment</th><th>status</th><th>model</th><th>lr</th><th>batch</th><th>epochs</th><th>train subset</th><th>input</th><th>val " + esc(metric) + "</th><th>min</th><th>hypothesis</th></tr>"
    parts.append("<h3>Ranking of completed full-validation runs</h3><table>" + header + "".join(row(r) for r in ranked) + "</table>")
    if others:
        parts.append("<h3>Screening, failed, interrupted and partial runs (never ranked with full runs)</h3><table>" + header + "".join(row(r, "screen" if r.get("screening") else ("failed" if r["status"] in ("failed", "interrupted") else "")) for r in others) + "</table>")

    curve_series_val, curve_series_loss = [], []
    for i, r in enumerate((ranked or recs)[:6]):
        rows = d.curves(r["experiment_id"])
        color = COLORS[i % len(COLORS)]
        name = f"{r['run_config']['model']} lr={r['run_config']['lr']:g} bs={r['run_config']['batch_size']} [{r['experiment_id'][-6:]}]" + (" screen" if r.get("screening") else "")
        curve_series_val.append({"name": name, "color": color, "points": [(c["epoch"], c.get(f"val_{metric}")) for c in rows]})
        curve_series_loss.append({"name": name, "color": color, "points": [(c["epoch"], c.get("train_loss")) for c in rows]})
    parts.append("<h3>Learning curves</h3><div class='grid'><div>" + line_chart(curve_series_val, y_label=f"val {metric}") + "</div><div>" + line_chart(curve_series_loss, y_label="train loss") + "</div></div>")

    if best:
        rec = d.store.read(best_id)
        ck = rec.get("checkpoint", {})
        parts.append(
            "<h3>Selected checkpoint</h3><table>"
            f"<tr><th>experiment</th><td>{esc(best_id)} (selected by {esc(best['selected_by'])}: {esc(best.get('rationale', ''))})</td></tr>"
            f"<tr><th>validation {esc(metric)}</th><td>{fmt(best['best_val_metric'])} at epoch {esc(rec['result'].get('best_epoch'))}</td></tr>"
            f"<tr><th>checkpoint</th><td>{esc(ck.get('best_path'))}<br><span class='muted'>sha256 {esc(ck.get('best_sha256'))}</span></td></tr>"
            f"<tr><th>normalization</th><td>{esc(rec.get('normalization', {}).get('path'))} <span class='muted'>hash {esc((rec.get('normalization', {}).get('stats_hash') or '')[:16])}, fitted on train split only</span></td></tr>"
            f"<tr><th>model</th><td>{esc(rec['model']['name'])}, {esc(rec['model']['in_channels'])} channels, {esc(rec['model']['n_classes'])} classes, {esc(rec['model']['parameters'])} parameters, pretrained: {esc(rec.get('pretrained') or 'none (from scratch)')}</td></tr>"
            f"<tr><th>seed / precision / device</th><td>{esc(rec['seed'])} / amp={esc(rec.get('precision', {}).get('use_amp'))} / {esc(rec.get('device'))} {esc((rec.get('environment', {}).get('gpu') or {}).get('name', ''))}</td></tr>"
            "</table>"
        )
        val = rec["result"].get("val_metrics_at_best") or {}
        if val.get("per_class"):
            if rec["task_type"] == "classification":
                parts.append("<details><summary>Per-class validation diagnostics (precision / recall / F1)</summary><table><tr><th>class</th><th>support</th><th>precision</th><th>recall</th><th>f1</th></tr>" + "".join(f"<tr><td>{esc(c['class'])}</td><td>{esc(c['support'])}</td><td>{fmt(c['precision'], 3)}</td><td>{fmt(c['recall'], 3)}</td><td>{fmt(c['f1'], 3)}</td></tr>" for c in val["per_class"]) + f"</table><div class='muted'>macro-F1 {fmt(val.get('macro_f1'))} (diagnostic)</div></details>")
            else:
                extra = f"; mIoU excluding {esc(val['miou_excluding']['classes'])}: {fmt(val['miou_excluding']['value'])}" if val.get("miou_excluding") else ""
                parts.append("<details><summary>Per-class validation IoU</summary><table><tr><th>class</th><th>target pixels</th><th>predicted pixels</th><th>IoU</th></tr>" + "".join(f"<tr><td>{esc(c['class'])}</td><td>{esc(c['target_pixels'])}</td><td>{esc(c['predicted_pixels'])}</td><td>{fmt(c['iou'], 3) if c['present'] else '<span class=muted>absent</span>'}</td></tr>" for c in val["per_class"]) + f"</table><div class='muted'>pixel accuracy {fmt(val.get('pixel_accuracy'))} is shown for completeness only (misleading on imbalanced masks){extra}</div></details>")

        ex = d.examples(rec)
        if ex is not None:
            cards = []
            names = rec["model"]["class_names"]
            for i in range(len(ex["rgb"])):
                rgb = ex["rgb"][i]
                scale = max(1, 96 // max(1, rgb.shape[0]))
                if rec["task_type"] == "classification":
                    y, p = int(ex["label"][i]), int(ex["pred"][i])
                    ok = "ok" if y == p else "bad"
                    cards.append(f"<div class='ex'><img src='{png_data_uri(rgb, scale)}'><br>label: {esc(names[y] if y < len(names) else y)}<br>pred: {esc(names[p] if p < len(names) else p)} {badge('correct' if y == p else 'wrong', ok)}</div>")
                else:
                    cards.append(f"<div class='ex'><div class='seg'><img src='{png_data_uri(rgb, scale)}'><img src='{png_data_uri(colour_mask(ex['label'][i]), scale)}'><img src='{png_data_uri(colour_mask(ex['pred'][i]), scale)}'></div><br>valid sample {esc(int(ex['indices'][i]))}: image | label | prediction</div>")
            legend = "".join(f"<span class='badge' style='background:rgb({c[0]},{c[1]},{c[2]});color:#fff'>{esc(n)}</span>" for n, c in zip(names, colour_mask(np.arange(len(names))))) if rec["task_type"] == "segmentation" else ""
            parts.append(f"<h3>Fixed validation examples (first {len(cards)} samples of the validation split; display contrast stretch only)</h3>{legend}<div class='examples'>{''.join(cards)}</div>")

    total_min = sum(float((r.get("compute") or {}).get("wall_time_seconds") or 0) for r in recs) / 60.0
    parts.append(f"<h3>Compute</h3><div class='muted'>{len(recs)} experiments, {total_min:.1f} wall-clock minutes in total (includes resumed time); device {esc(recs[-1].get('device'))}.</div>")
    parts.append("</section>")
    return "".join(parts)


def render_agent(d: DashboardData) -> str:
    if not d.sessions:
        return "<section><h2>Agent decisions</h2><div class='muted'>No agent sessions recorded.</div></section>"
    parts = ["<section><h2>Agent decisions</h2>"]
    for s in d.sessions:
        decisions = list(iter_jsonl(session_dir(d.home, s["session_id"]) / "decisions.jsonl"))
        parts.append(f"<h3>Session {esc(s['session_id'])} <span class='muted'>{esc(s.get('status'))}, tasks {esc(s.get('tasks'))}, model {esc(s.get('model_id') or 'n/a')}, {esc(s.get('tool_calls'))} tool calls, {esc(s.get('experiments_launched'))} experiments, {esc(round(float(s.get('elapsed_seconds', 0)) / 60, 1))} min</span></h3>")
        items = []
        for e in decisions:
            k = e["kind"]
            if k == "propose_experiment":
                c = e["config"]
                items.append(f"<div class='decision d-propose'><b>propose</b> {esc(c['model'])} lr={c['lr']:g} bs={c['batch_size']} epochs={c['epochs']} max={c['max_minutes']}min{' screening ' + str(c['subset_fraction']) if c.get('screening') else ''}<br><b>hypothesis:</b> {esc(c.get('hypothesis'))}<br><b>rationale:</b> {esc(c.get('rationale'))}<br><b>expected tradeoff:</b> {esc(c.get('expected_tradeoff'))}</div>")
            elif k == "experiment_result":
                items.append(f"<div class='decision d-result'><b>result</b> {esc(e.get('experiment_id'))}: {esc(e.get('status'))}, metric {fmt(e.get('best_val_metric'))}, stop {esc(e.get('stop_reason'))}{(' adjustments ' + esc(e.get('adjustments'))) if e.get('adjustments') else ''}{(' error: ' + esc(e.get('error'))) if e.get('error') else ''}</div>")
            elif k in ("guard_block", "rejected_experiment"):
                items.append(f"<div class='decision d-block'><b>{esc(k)}</b> {esc(e.get('tool', ''))}: {esc('; '.join(e.get('problems', [])))}</div>")
            elif k in ("select_best", "finish_task", "promote_model_version", "store_model_version", "verify_model_version"):
                items.append(f"<div class='decision d-select'><b>{esc(k)}</b> {esc({kk: vv for kk, vv in e.items() if kk not in ('at', 'session_id', 'kind')})}</div>")
            elif k == "round_end" and e.get("text"):
                items.append(f"<div class='decision'><b>round {esc(e.get('round'))} ({esc(e.get('task'))})</b> {esc(e.get('tool_calls'))} tool calls: {esc(e['text'][:600])}</div>")
            elif k in ("budget_exhausted", "idle_stop", "task_unfinished", "round_error", "session_end", "note"):
                items.append(f"<div class='decision'><b>{esc(k)}</b> {esc({kk: vv for kk, vv in e.items() if kk not in ('at', 'session_id', 'kind')})}</div>")
        parts.append("".join(items) or "<div class='muted'>no decisions</div>")
    parts.append("</section>")
    return "".join(parts)


def render_registry(d: DashboardData) -> str:
    rows = d.registry.list_versions()
    if not rows:
        return "<section><h2>Model registry</h2><div class='muted'>No model versions stored.</div></section>"
    parts = ["<section><h2>Model registry (filesystem backend)</h2><table><tr><th>version</th><th>task</th><th>experiment</th><th>metric (validation)</th><th>architecture</th><th>train min</th><th>parent</th><th>promoted</th><th>verification</th><th>sha256</th><th>note</th></tr>"]
    for v in rows:
        ver = v["verification"]
        full = d.registry.get_version(v["version_id"]).get("verification") or {}
        if ver is True and full.get("preprocessing_verified"):
            vb = badge("verified: model and preprocessing reproduced", "ok")
        elif ver is True:
            vb = badge("verified: stored reference batch reproduced", "ok")
        elif ver is False:
            vb = badge("verification FAILED", "bad")
        else:
            vb = badge("not verified", "grey")
        parts.append(f"<tr class='{'best' if v['promoted'] else ''}'><td>{esc(v['version_id'])}</td><td>{esc(v['task'])}</td><td>{esc(v['experiment_id'])}</td><td>{esc(v['metric'])} {fmt(v['value'])}</td><td>{esc(v['architecture'])}</td><td>{fmt((v['training_wall_time_seconds'] or 0) / 60, 1)}</td><td>{esc(v['parent_version'] or '')}</td><td>{badge('promoted', 'ok') if v['promoted'] else ''}</td><td>{vb}</td><td class='muted'>{esc(v['checkpoint_sha256'][:16])}</td><td>{esc(v['note'])}</td></tr>")
    parts.append("</table>")
    for v in rows:
        m = d.registry.get_version(v["version_id"])
        ver = m.get("verification")
        if ver:
            parts.append(f"<details><summary>{esc(v['version_id'])} verification detail</summary><pre>{esc({k: ver[k] for k in ver if k in ('verified', 'model_reproduced', 'preprocessing_verified', 'sha256_ok', 'max_abs_diff', 'mean_abs_diff', 'argmax_agreement', 'tolerance', 'n_reference', 'preprocessing_identity', 'note', 'error', 'at')})}</pre></details>")
    parts.append("<div class='muted'>store_model copies the checkpoint and a fixed validation batch with its logits; verify rebuilds the model from the manifest, checks the SHA-256 and reproduces those logits within the recorded tolerance. Versions are immutable; promotion only moves a pointer.</div></section>")
    return "".join(parts)


def render_final(d: DashboardData) -> str:
    parts = ["<section><h2>Frozen selection and final test evaluation</h2>"]
    if d.manifest:
        parts.append(f"<div>Selection frozen at {esc(d.manifest['frozen_at'])} by {esc(d.manifest.get('frozen_by'))}, manifest hash {esc(d.manifest['manifest_hash'][:16])}</div><table><tr><th>task</th><th>experiment</th><th>validation metric</th><th>checkpoint sha256</th><th>model version</th><th>selected by</th></tr>")
        for t, e in d.manifest["tasks"].items():
            parts.append(f"<tr><td>{esc(t)}</td><td>{esc(e['experiment_id'])}</td><td>{esc(e['metric']['name'])} {fmt(e['best_val_metric'])}</td><td class='muted'>{esc(e['checkpoint_sha256'][:16])}</td><td>{esc(e.get('model_version'))}</td><td>{esc(e['selected_by'])}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<div class='muted'>No frozen selection yet (python -m gotrain freeze).</div>")
    if d.final:
        parts.append("<h3>Test results (each task evaluated exactly once)</h3><table><tr><th>task</th><th>status</th><th>test metric</th><th>validation at selection</th><th>samples</th><th>evaluated at</th><th>error</th></tr>")
        for t, r in d.final["tasks"].items():
            parts.append(f"<tr class='{'failed' if r.get('status') != 'completed' else ''}'><td>{esc(t)}</td><td>{esc(r.get('status'))}</td><td><b>{fmt(r.get('test_metric'))}</b> {esc((r.get('metric') or {}).get('name', ''))}</td><td>{fmt(r.get('val_metric_at_selection'))}</td><td>{esc(r.get('n_test_samples'))}</td><td>{esc(r.get('evaluated_at'))}</td><td>{esc(r.get('error'))}</td></tr>")
        parts.append("</table>")
        exposures = list(iter_jsonl(d.home.final_eval / "exposure.jsonl"))
        parts.append(f"<div class='muted'>Test split openings logged: {len(exposures)} (final_eval/exposure.jsonl).</div>")
    else:
        parts.append("<div class='muted'>Final evaluation not run. The dashboard never loads test data itself.</div>")
    parts.append("</section>")
    return "".join(parts)


def render_reviews(d: DashboardData) -> str:
    root = d.home.root / "review"
    found = []
    if root.exists():
        for prov in sorted(root.glob("*/*/provenance.json")):
            try:
                p = read_json(prov)
            except ValueError:
                continue
            page = prov.parent / "review.html"
            found.append((p, page))
    if not found:
        return ""
    rows = []
    for p, page in found:
        data = p.get("data", {})
        model = p.get("model", {})
        label = f"{data.get('n_images_reviewed')} of {data.get('n_images_in_split')} validation images" + (" (PARTIAL)" if data.get("partial") else "")
        link = f"<a href='{esc('../' + str(page.relative_to(d.home.reports.parent)))}'>open review page</a>" if page.exists() else "<span class='muted'>review.html not rendered</span>"
        rows.append(f"<tr><td>{esc(model.get('task'))}</td><td>{esc(model.get('experiment_id'))}</td><td>{esc(model.get('model_version') or '')}</td><td>{esc(label)}</td><td>{esc(p.get('built_at'))}</td><td>{link}</td></tr>")
    return "<section><h2>Review pages (human inspection queues)</h2><table><tr><th>task</th><th>experiment</th><th>version</th><th>coverage</th><th>built</th><th></th></tr>" + "".join(rows) + "</table><div class='muted'>Links are relative to the run home; the review page itself is self-contained.</div></section>"


def render_environment(d: DashboardData) -> str:
    env = d.records[-1].get("environment", {}) if d.records else {}
    return f"<section><h2>Environment of the latest record</h2><pre>{esc(env)}</pre></section>"


def render_dashboard(home: Home, out: Path | None = None) -> Path:
    d = DashboardData(home)
    out = Path(out) if out else home.reports / "dashboard.html"
    banner = ""
    if d.fixture:
        banner = "<div class='banner'>SYNTHETIC FIXTURE: every number on this page comes from synthetic data generated for demonstration. No real GEO-Bench scores are shown.</div>"
    elif any(r.get("source") == "synthetic" for r in d.records):
        banner = "<div class='banner'>Some records below come from synthetic smoke tests and are labelled as such.</div>"
    missing = [t for t in REQUIRED_TASKS if t in d.tasks and t not in {r["task"] for r in d.records}] if not d.fixture else []
    if missing:
        banner += f"<div class='banner'>Missing results for required tasks: {esc(', '.join(missing))}</div>"
    body = [
        f"<header><h1>GoTrain dashboard</h1><div class='sub'>generated {esc(utc_now_iso())} from {esc(home.root)} ({len(d.records)} experiment records, {len(d.sessions)} agent sessions)</div></header><main>",
        banner,
        render_overview(d),
    ]
    for task in d.tasks:
        body.append(render_task(d, task))
    body.append(render_agent(d))
    body.append(render_registry(d))
    body.append(render_final(d))
    body.append(render_reviews(d))
    body.append(render_environment(d))
    body.append("</main>")
    doc = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>GoTrain dashboard</title><link rel='icon' href='data:,'><style>{CSS}</style></head><body>{''.join(body)}</body></html>"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not d.fixture:
        from gotrain.report.friendly import upgrade_document
        doc = upgrade_document(doc)
    out.write_text(doc, encoding="utf-8")
    return out
