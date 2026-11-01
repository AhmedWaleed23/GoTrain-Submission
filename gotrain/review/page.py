"""Self-contained review page: one HTML file, vanilla JavaScript, images embedded as
data URIs, no external resources. Reads the artifacts written by build.py."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from gotrain.review.notes import DECISION_LABELS, NOTES_FORMAT, run_identity

CSS = """
:root{--bg:#f4f6f8;--card:#fff;--ink:#1b2430;--muted:#5f6b7a;--line:#dde3ea;--accent:#1f5fbf;--ok:#1d8a4e;--warn:#b7791f;--bad:#c53030}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.4}
header{background:#14213d;color:#fff;padding:14px 20px}header h1{margin:0;font-size:20px}header p{margin:4px 0 0;opacity:.85;font-size:14px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;background:#e8edf3;color:#334}
.badge.partial{background:#fde68a;color:#78350f}.badge.synthetic{background:#fecaca;color:#7f1d1d}
main{display:grid;grid-template-columns:320px minmax(0,1fr);gap:16px;padding:16px;max-width:1500px;margin:0 auto}main>section{min-width:0}
@media (max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
h2{font-size:16px;margin:0 0 8px}h3{font-size:14px;margin:12px 0 6px}
.muted{color:var(--muted);font-size:13px}
#queue{max-height:70vh;overflow:auto;border-top:1px solid var(--line)}
.qitem{display:flex;gap:8px;align-items:center;padding:6px 4px;border-bottom:1px solid var(--line);cursor:pointer;font-size:13px}
.qitem:hover{background:#f0f4fa}.qitem.active{background:#e3ecfa;outline:2px solid var(--accent)}
.qitem .rank{width:34px;flex:none;font-weight:700;color:var(--accent)}.qitem .id{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dot{width:10px;height:10px;border-radius:50%;background:#cbd5e1;flex:none}.dot.looks_correct{background:var(--ok)}.dot.needs_correction{background:var(--bad)}.dot.unsure{background:var(--warn)}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.filters button{font-size:12px}
button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
button:hover{border-color:var(--accent)}button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
button.decide.looks_correct.on{background:var(--ok);border-color:var(--ok)}button.decide.needs_correction.on{background:var(--bad);border-color:var(--bad)}button.decide.unsure.on{background:var(--warn);border-color:var(--warn)}
.viewer{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.frame{position:relative;width:100%;aspect-ratio:1/1;background:#000;border-radius:6px;overflow:hidden}
.frame img{position:absolute;inset:0;width:100%;height:100%;image-rendering:auto}
.frame img.mask{image-rendering:pixelated;mix-blend-mode:normal}
.frame .cap{position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.55);color:#fff;font-size:12px;padding:3px 6px}
.toggles label{margin-right:12px;font-size:13px}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;margin-top:6px}.legend span{display:inline-flex;align-items:center;gap:4px}
.sw{width:14px;height:14px;border:1px solid #999;display:inline-block}
.decide-row{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0;min-width:0}.decide-row input,.decide-row label{max-width:100%}
textarea{width:100%;min-height:70px;font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px}
.nav{display:flex;gap:8px;align-items:center;margin:8px 0}
details{margin-top:12px}summary{cursor:pointer;font-weight:600}
table{border-collapse:collapse;font-size:13px;width:100%}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
.kv{font-size:12px;color:var(--muted);word-break:break-all}table{display:block;overflow-x:auto}
#status{font-size:13px;min-height:18px;color:var(--muted)}#status.err{color:var(--bad)}
.hidden{display:none}
kbd{background:#eef2f7;border:1px solid #cbd5e1;border-radius:4px;padding:0 4px;font-size:12px}
"""

JS = r"""
(function(){
'use strict';
const D = window.REVIEW_DATA;
const items = D.items; const byId = {}; items.forEach(it => byId[it.sample_id] = it);
const order = D.queue.slice();
const ident = D.identity; const storageKey = 'gotrain-review:' + ident.identity_hash;
let reviews = {}; let filter = 'all'; let current = order[0] || null;
const $ = id => document.getElementById(id);
function setStatus(msg, err){ const s = $('status'); s.textContent = msg || ''; s.className = err ? 'err' : ''; }
function loadLocal(){ try { const raw = localStorage.getItem(storageKey); if (raw) { const obj = JSON.parse(raw); if (obj && obj.identity_hash === ident.identity_hash) reviews = obj.reviews || {}; } } catch (e) { setStatus('Browser storage unavailable; use Export to keep your notes.'); } }
function saveLocal(){ try { localStorage.setItem(storageKey, JSON.stringify({identity_hash: ident.identity_hash, reviews: reviews})); } catch (e) { setStatus('Browser storage unavailable; export your notes before closing.'); } }
function decisionOf(id){ const r = reviews[id]; return r && r.decision ? r.decision : null; }
function visible(id){ const d = decisionOf(id); if (filter === 'all') return true; if (filter === 'unreviewed') return !d; return d === filter; }
function renderQueue(){
  const q = $('queue'); q.textContent = '';
  let shown = 0;
  order.forEach(id => { if (!visible(id)) return; shown++; const it = byId[id];
    const row = document.createElement('div'); row.className = 'qitem' + (id === current ? ' active' : ''); row.dataset.id = id; row.setAttribute('role','option'); row.tabIndex = 0;
    const rank = document.createElement('span'); rank.className = 'rank'; rank.textContent = '#' + it.rank;
    const dot = document.createElement('span'); dot.className = 'dot ' + (decisionOf(id) || ''); dot.title = decisionOf(id) ? D.labels[decisionOf(id)] : 'not reviewed';
    const name = document.createElement('span'); name.className = 'id'; name.textContent = id;
    const pr = document.createElement('span'); pr.className = 'muted'; pr.textContent = it.priority.toFixed(2);
    row.append(rank, dot, name, pr); row.addEventListener('click', () => select(id)); row.addEventListener('keydown', e => { if (e.key === 'Enter') select(id); });
    q.appendChild(row); });
  if (!shown) { const p = document.createElement('p'); p.className = 'muted'; p.textContent = 'No images match this filter.'; q.appendChild(p); }
  const reviewed = order.filter(id => decisionOf(id)).length;
  $('progress').textContent = reviewed + ' of ' + order.length + ' reviewed';
  const act = q.querySelector('.qitem.active'); if (act) act.scrollIntoView({block: 'nearest'});
}
function select(id){ current = id; renderQueue(); renderViewer(); }
function renderViewer(){
  if (!current) { $('viewer').classList.add('hidden'); $('empty').classList.remove('hidden'); return; }
  $('viewer').classList.remove('hidden'); $('empty').classList.add('hidden');
  const it = byId[current]; const img = D.images[it.stem];
  $('img-rgb').src = img.rgb; $('img-rgb2').src = img.rgb; $('img-rgb3').src = img.rgb;
  $('img-pred').src = img.pred; $('img-ent').src = img.ent; $('img-gt').src = img.gt;
  $('title').textContent = 'Image ' + it.rank + ' of ' + order.length + ' in review priority';
  $('sid').textContent = it.sample_id;
  $('facts').textContent = 'Review priority ' + it.priority.toFixed(2) + ' (0 = confident everywhere, 1 = very uncertain) · predicted ' + D.foreground_label + ' area ' + (100*it.predicted_foreground_fraction).toFixed(1) + '% of pixels · uncertain pixels ' + (100*it.uncertain_pixel_fraction).toFixed(1) + '%';
  applyToggles();
  const d = decisionOf(current);
  document.querySelectorAll('button.decide').forEach(b => b.classList.toggle('on', b.dataset.decision === d));
  const r = reviews[current] || {}; $('note').value = r.note || '';
  $('gt-facts').textContent = D.demo ? ('Validation demo: foreground IoU ' + (it.foreground_iou === null ? 'n/a' : it.foreground_iou.toFixed(3)) + (it.both_empty ? ' (no ' + D.foreground_label + ' in truth or prediction)' : '') + ', misclassified pixels ' + it.misclassified_pixels + ' of ' + it.pixels) : '';
}
function applyToggles(){ $('img-pred').style.display = $('t-pred').checked ? '' : 'none'; $('img-ent').style.display = $('t-ent').checked ? '' : 'none'; $('frame-gt').style.display = $('t-gt').checked ? '' : 'none'; $('img-gt').style.display = ''; }
function move(delta){ const vis = order.filter(visible); if (!vis.length) return; let i = vis.indexOf(current); i = i < 0 ? 0 : Math.min(vis.length - 1, Math.max(0, i + delta)); select(vis[i]); }
function decide(dec){ if (!current) return; const r = reviews[current] || {}; r.decision = (r.decision === dec) ? null : dec; r.updated_at = new Date().toISOString(); reviews[current] = r; saveLocal(); renderQueue(); renderViewer(); }
function noteChanged(){ if (!current) return; const r = reviews[current] || {}; r.note = $('note').value; r.updated_at = new Date().toISOString(); reviews[current] = r; saveLocal(); }
function exportPayload(){ const entries = Object.keys(reviews).sort().map(id => ({sample_id: id, decision: reviews[id].decision || null, note: reviews[id].note || '', updated_at: reviews[id].updated_at || null})).filter(e => e.decision || e.note);
  return {format: D.notes_format, exported_at: new Date().toISOString(), reviewer: $('reviewer').value || '', identity: ident, meaning: 'human review labels for inspection; not training annotations, corrected masks or certification', reviews: entries}; }
function download(name, text, type){ const blob = new Blob([text], {type: type}); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500); }
function csvCell(v){ return '"' + String(v === null || v === undefined ? '' : v).replace(/"/g, '""') + '"'; }
function toCsv(p){ const head = ['sample_id','decision','decision_label','note','updated_at','task','experiment_id','model_version','checkpoint_sha256','split','exported_at'];
  const rows = [head.map(csvCell).join(',')];
  p.reviews.forEach(e => rows.push([e.sample_id, e.decision || '', e.decision ? D.labels[e.decision] : '', e.note, e.updated_at || '', p.identity.task, p.identity.experiment_id, p.identity.model_version || '', p.identity.checkpoint_sha256, p.identity.split, p.exported_at].map(csvCell).join(','))); return rows.join('\n') + '\n'; }
function stamp(){ return new Date().toISOString().replace(/[:.]/g, '-'); }
function validateImport(obj){
  if (!obj || typeof obj !== 'object') throw new Error('notes file must be a JSON object');
  if (obj.format !== D.notes_format) throw new Error('unsupported notes format ' + obj.format);
  const id = obj.identity || {};
  ['identity_hash','checkpoint_sha256','experiment_id','split'].forEach(k => { if (id[k] !== ident[k]) throw new Error('notes were exported from a different run (' + k + ' differs)'); });
  if (!Array.isArray(obj.reviews)) throw new Error('reviews must be a list');
  obj.reviews.forEach((e, i) => { if (!e || typeof e !== 'object' || !('sample_id' in e)) throw new Error('entry ' + i + ' malformed'); if (e.decision && !(e.decision in D.labels)) throw new Error('entry ' + i + ' has unknown decision ' + e.decision); if (e.note !== undefined && typeof e.note !== 'string') throw new Error('entry ' + i + ' note must be text'); });
  return obj.reviews;
}
function importFile(file){ const reader = new FileReader(); reader.onload = () => { try { const entries = validateImport(JSON.parse(reader.result)); let n = 0; entries.forEach(e => { if (!(e.sample_id in byId)) return; reviews[e.sample_id] = {decision: e.decision || null, note: e.note || '', updated_at: e.updated_at || null}; n++; }); saveLocal(); renderQueue(); renderViewer(); setStatus('Imported ' + n + ' review entries.'); } catch (err) { setStatus('Import rejected: ' + err.message, true); } }; reader.readAsText(file); }
function renderBenchmark(){
  const b = D.benchmark; const tb = $('bench-rows'); tb.textContent = '';
  if (!b.comparison.applicable) { const tr = document.createElement('tr'); const td = document.createElement('td'); td.colSpan = 5; td.textContent = 'No misclassified pixels in the reviewed images: the error-capture benchmark is not applicable.'; tr.appendChild(td); tb.appendChild(tr); return; }
  Object.keys(b.comparison.budgets).forEach(k => { const r = b.comparison.budgets[k]; const tr = document.createElement('tr'); [k + ' (' + r.images + ' images)', (100*r.ranked_proportion).toFixed(1) + '%', (100*r.random_mean).toFixed(1) + '%', (100*r.random_p5).toFixed(1) + '% to ' + (100*r.random_p95).toFixed(1) + '%', r.above_random_p95 ? 'above random p95' : (r.beats_random_mean ? 'above random mean' : 'not above random mean')].forEach(v => { const td = document.createElement('td'); td.textContent = v; tr.appendChild(td); }); tb.appendChild(tr); });
}
function init(){
  loadLocal();
  document.querySelectorAll('.filters button').forEach(b => b.addEventListener('click', () => { filter = b.dataset.filter; document.querySelectorAll('.filters button').forEach(x => x.classList.toggle('on', x === b)); renderQueue(); }));
  ['t-pred','t-ent','t-gt'].forEach(id => $(id).addEventListener('change', applyToggles));
  document.querySelectorAll('button.decide').forEach(b => b.addEventListener('click', () => decide(b.dataset.decision)));
  $('note').addEventListener('input', noteChanged);
  $('prev').addEventListener('click', () => move(-1)); $('next').addEventListener('click', () => move(1));
  $('export-json').addEventListener('click', () => { const p = exportPayload(); download('review-notes-' + ident.task + '-' + stamp() + '.json', JSON.stringify(p, null, 2), 'application/json'); setStatus('Exported ' + p.reviews.length + ' review entries as JSON.'); });
  $('export-csv').addEventListener('click', () => { const p = exportPayload(); download('review-notes-' + ident.task + '-' + stamp() + '.csv', toCsv(p), 'text/csv'); setStatus('Exported ' + p.reviews.length + ' review entries as CSV.'); });
  $('import-file').addEventListener('change', e => { if (e.target.files[0]) importFile(e.target.files[0]); e.target.value = ''; });
  $('clear-local').addEventListener('click', () => { if (confirm('Remove all review decisions and notes stored in this browser for this model? Exported files are not affected.')) { reviews = {}; saveLocal(); renderQueue(); renderViewer(); setStatus('Browser-stored notes cleared.'); } });
  document.addEventListener('keydown', e => { if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return; if (e.key === 'ArrowDown' || e.key === 'j') { move(1); e.preventDefault(); } else if (e.key === 'ArrowUp' || e.key === 'k') { move(-1); e.preventDefault(); } else if (e.key === '1') decide('looks_correct'); else if (e.key === '2') decide('needs_correction'); else if (e.key === '3') decide('unsure'); else if (e.key === 'p') { $('t-pred').checked = !$('t-pred').checked; applyToggles(); } else if (e.key === 'u') { $('t-ent').checked = !$('t-ent').checked; applyToggles(); } else if (e.key === 'g' && D.demo) { $('t-gt').checked = !$('t-gt').checked; applyToggles(); } });
  renderBenchmark(); renderQueue(); renderViewer();
}
document.addEventListener('DOMContentLoaded', init);
})();
"""


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _mask_rgba_png(gray_png_path: Path, color: tuple[int, int, int], alpha: int) -> str:
    """Turn a grayscale mask/heat PNG into a coloured RGBA overlay data URI."""
    try:
        from PIL import Image
        import io

        im = Image.open(gray_png_path).convert("L")
        import numpy as np

        arr = np.asarray(im, dtype=np.uint8)
        rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
        rgba[..., 3] = (arr.astype(np.uint16) * alpha // 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        return _data_uri(gray_png_path, "image/png")


def _heat_rgba_png(gray_png_path: Path) -> str:
    """Entropy heat map: transparent where certain, yellow to red where uncertain."""
    try:
        from PIL import Image
        import io
        import numpy as np

        arr = np.asarray(Image.open(gray_png_path).convert("L"), dtype=np.float32) / 255.0
        rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = (255 * (1 - arr)).astype(np.uint8)
        rgba[..., 2] = 0
        rgba[..., 3] = (np.clip(arr, 0, 1) * 200).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        return _data_uri(gray_png_path, "image/png")


def review_presentation(provenance: dict) -> dict:
    task = provenance["model"]["task"]
    n_classes = provenance["model"]["architecture"]["n_classes"]
    # The existing synthetic display fixture has three object/background classes.
    # It is explicitly labelled as synthetic; retain that test-only rendering path.
    if n_classes != 2 and task != "synthetic":
        raise ValueError("Interactive overlays currently support binary segmentation only; use the technical report for multiclass results")
    if task in ("m-pv4ger-seg", "synthetic"):
        return {"title": "Review solar-panel predictions", "regions": "solar regions", "foreground": "solar", "domain": "on this solar-panel validation dataset"}
    if task == "m-nz-cattle":
        return {"title": "Review cattle predictions", "regions": "cattle regions", "foreground": "cattle", "domain": "on this cattle validation dataset"}
    return {"title": "Review segmentation predictions", "regions": "foreground regions", "foreground": "foreground", "domain": "on this validation dataset"}


def render_review_page(review: dict, out_path: Path | None = None, max_embedded: int | None = None) -> Path:
    prov = review["provenance"]
    presentation = review_presentation(prov)
    items = review["items"]
    cache = Path(review["dir"]) / "cache"
    fmt = prov["thumbnails"]["rgb_format"]
    rgb_mime = "image/jpeg" if fmt == "jpeg" else "image/png"
    embedded_items = items if max_embedded is None else items[:max_embedded]
    images = {}
    for it in embedded_items:
        stem = it["stem"]
        images[stem] = {
            "rgb": _data_uri(cache / f"{stem}_rgb.{fmt}", rgb_mime),
            "pred": _mask_rgba_png(cache / f"{stem}_pred.png", (0, 170, 255), 150),
            "gt": _mask_rgba_png(cache / f"{stem}_gt.png", (60, 220, 90), 150),
            "ent": _heat_rgba_png(cache / f"{stem}_ent.png"),
        }
    embedded_ids = {it["sample_id"] for it in embedded_items}
    queue = [sid for sid in review["queue"] if sid in embedded_ids]
    identity = run_identity(prov)
    data = {
        "identity": identity,
        "items": [{k: v for k, v in it.items()} for it in embedded_items],
        "queue": queue,
        "images": images,
        "benchmark": review["benchmark"],
        "labels": DECISION_LABELS,
        "notes_format": NOTES_FORMAT,
        "demo": True,
        "foreground_label": presentation["foreground"],
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    task = prov["model"]["task"]
    n_rev = prov["data"]["n_images_reviewed"]
    n_split = prov["data"]["n_images_in_split"]
    partial = prov["data"]["partial"]
    synthetic = prov.get("synthetic")
    comp = review["benchmark"]["comparison"]
    model = prov["model"]
    badges = ""
    if partial:
        badges += f"<span class='badge partial'>PARTIAL: {n_rev} of {n_split} validation images</span> "
    if synthetic:
        badges += "<span class='badge synthetic'>SYNTHETIC FIXTURE: not real results</span> "
    if max_embedded is not None and len(embedded_items) < len(items):
        badges += f"<span class='badge partial'>only {len(embedded_items)} of {len(items)} images embedded in this page</span> "
    vm = model.get("validation_metric") or {}
    diag = vm.get("diagnostics") or {}
    bench_note = "Benchmark not applicable: no misclassified pixels." if not comp["applicable"] else ""
    curve = review["benchmark"].get("cumulative_capture_curve")
    curve_svg = ""
    if curve:
        n = len(curve)
        w, h, ml, mb = 420, 200, 36, 24
        pts = " ".join(f"{ml + (i / max(1, n - 1)) * (w - ml - 8):.1f},{8 + (1 - v) * (h - mb - 8):.1f}" for i, v in enumerate(curve))
        diag_line = f"{ml},{h - mb} {w - 8},8"
        curve_svg = (
            f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:420px' xmlns='http://www.w3.org/2000/svg' role='img' aria-label='cumulative error capture'>"
            f"<rect x='{ml}' y='8' width='{w - ml - 8}' height='{h - mb - 8}' fill='#fff' stroke='#ccc'/>"
            f"<polyline points='{diag_line}' fill='none' stroke='#999' stroke-dasharray='4 3'/>"
            f"<polyline points='{pts}' fill='none' stroke='#1f5fbf' stroke-width='2'/>"
            f"<text x='{ml}' y='{h - 6}' font-size='10'>images reviewed in priority order (0 to {n})</text>"
            f"<text x='4' y='14' font-size='10'>share of all misclassified pixels</text></svg>"
            "<div class='muted'>blue: priority order; dashed: what a perfectly uniform random order would capture on average</div>"
        )
    ref = model.get("reference_check") or {}
    mc = prov.get("metric_comparison") or {}
    doc = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(presentation["title"])}</title><link rel='icon' href='data:,'><style>{CSS}</style></head><body>
<header><h1>{html.escape(presentation["title"])}</h1><p>Images are ordered so the predictions the model was least sure about come first; decide for each whether the predicted {html.escape(presentation["regions"])} look right. {badges}</p></header>
<main>
<section class='panel' aria-label='review queue'>
<h2>Review queue</h2><div id='progress' class='muted'></div>
<div class='filters' role='group' aria-label='filters'><button data-filter='all' class='on'>All</button><button data-filter='unreviewed'>Unreviewed</button><button data-filter='needs_correction'>Needs correction</button><button data-filter='looks_correct'>Looks correct</button><button data-filter='unsure'>Unsure</button></div>
<div id='queue' role='listbox' aria-label='images by review priority'></div>
<p class='muted'>Keys: <kbd>j</kbd>/<kbd>k</kbd> or arrows move, <kbd>1</kbd> looks correct, <kbd>2</kbd> needs correction, <kbd>3</kbd> unsure, <kbd>p</kbd>/<kbd>u</kbd>/<kbd>g</kbd> toggle overlays.</p>
</section>
<section class='panel' aria-label='selected image'>
<p id='empty' class='muted hidden'>No images to show.</p>
<div id='viewer'>
<h2 id='title'></h2><div class='kv'>sample <span id='sid'></span></div><div id='facts' class='muted'></div>
<div class='nav'><button id='prev'>Previous</button><button id='next'>Next</button><span id='status' role='status' aria-live='polite'></span></div>
<div class='toggles'><label><input type='checkbox' id='t-pred' checked> Predicted {html.escape(presentation["regions"])}</label><label><input type='checkbox' id='t-ent' checked> Uncertainty overlay</label><label><input type='checkbox' id='t-gt'> Show correct mask (validation demo)</label></div>
<div class='viewer'>
<div><div class='frame'><img id='img-rgb' alt='original aerial image'><span class='cap'>Original image</span></div></div>
<div><div class='frame'><img id='img-rgb2' alt=''><img id='img-pred' class='mask' alt='predicted {html.escape(presentation["regions"])} overlay'><span class='cap'>Predicted {html.escape(presentation["regions"])} (blue)</span></div></div>
<div><div class='frame'><img id='img-rgb3' alt=''><img id='img-ent' class='mask' alt='uncertainty overlay'><span class='cap'>Where the model was unsure (yellow to red)</span></div></div>
<div id='frame-gt' style='display:none'><div class='frame'><img id='img-gt' class='mask' alt='correct mask'><span class='cap'>Correct mask from the validation set (green)</span></div><div id='gt-facts' class='muted'></div></div>
</div>
<div class='legend'><span><i class='sw' style='background:rgba(0,170,255,.6)'></i> predicted {html.escape(presentation["foreground"])}</span><span><i class='sw' style='background:rgba(255,255,0,.6)'></i> somewhat unsure</span><span><i class='sw' style='background:rgba(255,0,0,.8)'></i> very unsure</span><span><i class='sw' style='background:rgba(60,220,90,.6)'></i> correct mask (demo)</span></div>
<h3>Your review</h3>
<div class='decide-row'><button class='decide looks_correct' data-decision='looks_correct'>Looks correct</button><button class='decide needs_correction' data-decision='needs_correction'>Needs correction</button><button class='decide unsure' data-decision='unsure'>Unsure</button></div>
<textarea id='note' placeholder='Notes for this image (optional)' aria-label='review note'></textarea>
<p class='muted'>These are human review labels to direct inspection. They are not training annotations, corrected masks or a certification, and nothing is retrained or promoted automatically.</p>
<h3>Save your work</h3>
<div class='decide-row'><input id='reviewer' placeholder='Reviewer name (optional)' aria-label='reviewer name' style='font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px'><button id='export-json'>Export notes (JSON)</button><button id='export-csv'>Export notes (CSV)</button><label>Import notes JSON <input type='file' id='import-file' accept='application/json'></label><button id='clear-local'>Clear browser copy</button></div>
<p class='muted'>Decisions are kept in this browser for this model only; the exported file is the durable copy and can be re-imported here.</p>
</div>
<details><summary>Technical evidence and ranking benchmark</summary>
<p class='muted'>Ranking signal: {html.escape(review['benchmark']['signal']['pixel_signal'])}; per image, {html.escape(review['benchmark']['signal']['image_aggregate'])}; ties by sample id. The signal never sees the correct masks. It is an uncertainty heuristic, not calibrated correctness and not an explanation of the model's reasoning.</p>
<h3>Does the order help find mistakes? (retrospective check on the validation set)</h3>
<p class='muted'>Share of all misclassified pixels contained in the first 10, 20 and 30 percent of images, priority order versus 200 uniformly random orders (seed 42). Random bounds are simulation percentiles, not a confidence interval. {html.escape(bench_note)} This is exploratory evidence on this dataset, not a deployment guarantee or a measurement of reviewer time saved.</p>
<table><thead><tr><th>review budget</th><th>priority order</th><th>random mean</th><th>random p5 to p95</th><th>reading</th></tr></thead><tbody id='bench-rows'></tbody></table>
{curve_svg}
<h3>Model and data</h3>
<table><tbody>
<tr><th>task / split</th><td>{html.escape(task)} / {html.escape(prov['data']['split'])} ({n_rev} of {n_split} images{', PARTIAL' if partial else ''})</td></tr>
<tr><th>experiment</th><td>{html.escape(model['experiment_id'])} (selected by {html.escape(str(model.get('selected_by') or 'n/a'))})</td></tr>
<tr><th>model version</th><td>{html.escape(str(model.get('model_version') or 'none'))}</td></tr>
<tr><th>architecture</th><td>{html.escape(str(model['architecture'].get('name')))}, input size {html.escape(str(model.get('input_size') or 'native'))}, {html.escape(str(model['architecture'].get('parameters')))} parameters</td></tr>
<tr><th>recorded validation</th><td>{html.escape(str(vm.get('name')))} {html.escape(f"{vm.get('value'):.4f}" if isinstance(vm.get('value'), (int, float)) else 'n/a')}{html.escape(', foreground IoU ' + f"{diag['foreground_iou']:.4f}" if isinstance(diag.get('foreground_iou'), (int, float)) else '')}</td></tr>
<tr><th>review recomputation (fp32)</th><td>{html.escape(f"mIoU {prov['split_metrics_fp32']['miou']:.4f}, pixel accuracy {prov['split_metrics_fp32']['pixel_accuracy']:.4f}")}{html.escape(f"; difference from recorded {mc['abs_diff']:.5f} ({'within' if mc.get('within_tolerance') else 'OUTSIDE'} tolerance {mc['tolerance']})" if mc else '; full-split comparison not applicable to a partial build')}</td></tr>
<tr><th>checksums</th><td class='kv'>checkpoint {html.escape(model['checkpoint_sha256'])}<br>normalization {html.escape(model['normalization_hash'])}</td></tr>
<tr><th>reference check</th><td>{html.escape(f"max abs diff {ref['max_abs_diff']:.2e} on {ref['n']} stored inputs, within tolerance: {ref['within_tolerance']}" if ref else 'not run')}</td></tr>
<tr><th>inference</th><td>{html.escape(prov['inference']['device'])}, fp32, no augmentation, logits upsampled to native {html.escape(str(prov['data'].get('native_mask_size')))} masks, {html.escape(str(prov['inference']['seconds']))}s</td></tr>
<tr><th>both-empty images</th><td>{review['benchmark']['both_empty_images']} (foreground IoU counted as 1.0)</td></tr>
<tr><th>built</th><td>{html.escape(prov['built_at'])}</td></tr>
</tbody></table>
<h3>Limits of this page</h3><p class='muted'>Uncertainty is where the model's own output is close to a coin flip; a confident wrong prediction will not be highlighted. The benchmark reuses the validation set that also informed model selection, {html.escape(presentation["domain"])}; it says nothing about other regions, rooftop suitability or energy yield. Sample ids are the dataset's own file names; no map coordinates are derived or displayed.</p>
</details>
</section>
</main>
<script>window.REVIEW_DATA = {payload};</script>
<script>{JS}</script>
</body></html>"""
    out_path = Path(out_path) if out_path else Path(review["dir"]) / "review.html"
    out_path.write_text(doc, encoding="utf-8")
    return out_path
