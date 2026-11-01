"""Presentation layer for GoTrain's existing evidence report. Standard library only."""
import html
import re

TASKS = {
    'm-eurosat': ('Land-use classification', 'EuroSAT', 'Classifies a satellite image into a land-use category.', 'accuracy'),
    'm-pv4ger': ('Solar-panel detection', 'PV4GER', 'Classifies whether an aerial image contains solar panels.', 'accuracy'),
    'm-pv4ger-seg': ('Solar-panel mapping', 'PV4GER Segmentation', 'Outlines the pixels occupied by solar panels.', 'mean IoU'),
    'm-nz-cattle': ('Cattle mapping', 'New Zealand Cattle', 'Marks cattle locations in aerial images.', 'mean IoU'),
    'm-chesapeake': ('Land-cover mapping', 'Chesapeake', 'Assigns a land-cover label to each pixel.', 'mean IoU'),
}


def text(s):
    return html.unescape(re.sub(r'<[^>]*>', ' ', s)).strip()


def esc(s):
    return html.escape(str(s), quote=True)


def pct(value):
    return f'{float(value) * 100:.1f}%'


CSS = '''
:root{--ink:#182d2b;--green:#116b56;--mint:#e6f4ed;--line:#dce5e0;--paper:#f5f7f3;--muted:#526762}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
a{color:var(--green)}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #2879bd;outline-offset:4px}
.shell{max-width:1500px;margin:auto;display:grid;grid-template-columns:210px minmax(0,1fr);min-height:100vh}.rail{padding:34px 22px;border-right:1px solid var(--line);background:#fff}.brand{font-size:25px;font-weight:800;letter-spacing:-1px}.brand em{color:var(--green);font-style:normal}.rail small{display:block;color:var(--muted);margin:5px 0 34px}.rail nav{display:grid;gap:10px;position:sticky;top:24px}.rail a{text-decoration:none;padding:10px;border-radius:8px;color:var(--ink)}.rail a:hover{background:var(--mint)}
.workspace{padding:32px 38px;min-width:0}.topline{display:flex;align-items:center;justify-content:space-between;gap:12px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:1.7px;font-weight:750;color:var(--green)}.print{border:1px solid var(--line);background:white;padding:8px 14px;border-radius:8px}h1{font-size:34px;line-height:1.15;letter-spacing:-1.3px;margin:13px 0}h2{font-size:23px;letter-spacing:-.5px;margin:0 0 12px}h3{font-size:17px;margin:24px 0 10px}p{margin:8px 0}.intro{color:var(--muted);max-width:740px}.notice{margin:24px 0;padding:16px 20px;background:var(--mint);border-left:4px solid var(--green);border-radius:7px}.notice p{margin:4px 0;color:#31564c}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:24px 0}.task-card{text-align:left;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:12px;padding:19px;min-width:0;transition:box-shadow .15s}.task-card:hover,.task-card[aria-pressed=true]{border-color:var(--green);box-shadow:0 0 0 1px var(--green)}.task-card .task-name{display:block;font-weight:650;min-height:48px;font-size:14px}.task-card .score{display:block;font-size:32px;font-weight:760;letter-spacing:-1px;margin:10px 0 0}.task-card .score-label{font-size:12px;color:var(--muted)}.task-card .dataset{display:block;font-size:11px;color:var(--green);margin-top:12px;font-weight:650}.section-label{display:flex;align-items:center;justify-content:space-between;margin:30px 0 12px}.section-label span{color:var(--muted);font-size:12px}.panel,.evidence,.glossary{background:#fff;border:1px solid var(--line);border-radius:12px;padding:25px;margin-bottom:20px;min-width:0}.panel-head{display:flex;justify-content:space-between;gap:18px;align-items:start}.tag{background:var(--mint);color:var(--green);padding:5px 9px;border-radius:5px;font-size:11px;font-weight:700;white-space:nowrap}.panel .muted,.muted{color:var(--muted);font-size:13px}.limitation{padding:13px 16px;margin:18px 0;background:#fff7e7;border:1px solid #f0dfb9;border-radius:8px;font-size:13px}.examples{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.ex{background:#f7f9f6;border:1px solid var(--line);border-radius:9px;padding:12px;font-size:12px;overflow:hidden}.ex>img{display:block;max-width:100%;height:auto;margin-bottom:8px;width:160px}.seg{display:flex;gap:5px}.seg img{width:calc((100% - 10px)/3);height:auto;object-fit:contain;min-width:0;align-self:start}.badge{display:inline-block;padding:3px 7px;border-radius:5px;font-size:11px;margin:3px}.b-ok{background:#e4f4e9;color:#22573a}.b-warn{background:#fff1d4;color:#754500}.b-bad{background:#ffebe8;color:#922b22}.b-info{background:#e8f0fa;color:#254f83}.b-grey{background:#edf0ec;color:#526762}
details{margin:14px 0}summary{cursor:pointer;font-weight:650;padding:10px 0}details.technical{border-top:1px solid var(--line);margin-top:24px;padding-top:8px}.table-scroll{overflow:auto;max-width:100%}table{border-collapse:collapse;font-size:12px;width:100%;margin:10px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#f2f5f0;font-weight:650}td{overflow-wrap:anywhere}tr.best{background:#edf8ef}tr.screen{color:var(--muted)}pre{overflow:auto;background:#152e2b;color:#d5e9de;border-radius:8px;padding:16px;font-size:12px}.grid{display:flex;flex-wrap:wrap;gap:16px}.grid>div{min-width:0;max-width:100%}svg{max-width:100%;height:auto}svg text{font-size:10px;fill:#526762}.decision{border-left:3px solid #a3bcb0;padding:8px 12px;margin:8px 0;font-size:13px}.banner{padding:18px;background:#fff1cf;border:2px solid #b37a18;margin:16px 0;font-weight:700}.evidence>details>summary{font-size:16px}.glossary-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}.glossary-grid b{display:block}footer{font-size:12px;color:var(--muted);padding:12px 0 28px;overflow-wrap:anywhere}[hidden]{display:none!important}
@media(max-width:1100px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.workspace{padding:26px}.shell{grid-template-columns:175px minmax(0,1fr)}}@media(max-width:720px){.shell{display:block}.rail{padding:16px 20px;border-right:0;border-bottom:1px solid var(--line)}.rail small{display:none}.rail nav{display:flex;gap:4px;position:static;font-size:12px;margin-top:8px}.rail a{padding:6px}.workspace{padding:20px 15px}h1{font-size:29px}.task-card{padding:14px}.task-card .score{font-size:28px}.examples,.glossary-grid{grid-template-columns:1fr}.panel,.evidence,.glossary{padding:18px}.panel-head{display:block}.tag{display:inline-block;margin:8px 0}.section-label{display:block}.task-card .task-name{min-height:44px}}
@media print{.rail,.print{display:none}.shell{display:block}.workspace{padding:0}.panel[hidden]{display:block!important}.panel,.task-card{break-inside:avoid}.technical,.evidence{display:none}.cards{grid-template-columns:repeat(4,1fr)}}
'''

JS = '''
document.querySelectorAll('[data-task-button]').forEach(button=>button.addEventListener('click',()=>{
 const id=button.dataset.taskButton;
 document.querySelectorAll('[data-task-panel]').forEach(panel=>{panel.hidden=panel.dataset.taskPanel!==id;});
 document.querySelectorAll('[data-task-button]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
 document.getElementById('task-detail').scrollIntoView({behavior:'smooth',block:'start'});
}));
document.querySelectorAll('table').forEach(table=>{const wrap=document.createElement('div');wrap.className='table-scroll';table.parentNode.insertBefore(wrap,table);wrap.appendChild(table);});
document.getElementById('print-report').addEventListener('click',()=>window.print());
'''


def upgrade_document(doc):
    sections = re.findall(r'<section\b[^>]*>.*?</section>', doc, re.S)
    if not sections:
        raise ValueError('No report sections found; original report was left intact')
    overview = next((s for s in sections if '<h2>Task overview</h2>' in s), '')
    rows = {}
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', overview, re.S):
        cells = re.findall(r'<td\b[^>]*>(.*?)</td>', row, re.S)
        if len(cells) >= 8 and text(cells[0]) in TASKS:
            rows[text(cells[0])] = cells
    cards, panels = [], []
    for i, (task, (name, dataset, purpose, metric)) in enumerate((entry for entry in TASKS.items() if entry[0] in rows)):
        cells = rows.get(task)
        score_match = re.match(r'\s*(0(?:\.\d+)?|1(?:\.0+)?)\b', text(cells[4])) if cells else None
        score = pct(score_match.group(1)) if score_match else 'Pending'
        test_match = re.match(r'\s*(0(?:\.\d+)?|1(?:\.0+)?)\b', text(cells[7])) if cells else None
        display_score = pct(test_match.group(1)) if test_match else score
        score_label = 'Final test' if test_match else 'Validation'
        selected = score_match is not None
        section = next((s for s in sections if re.match(r'<section\b[^>]*><h2>' + re.escape(task) + r'(?:\s|<)', s)), '')
        cards.append(f'<button class="task-card" data-task-button="{task}" aria-pressed="{str(i == 0).lower()}" aria-controls="panel-{task}"><span class="task-name">{name}</span><span class="score">{display_score}</span><span class="score-label">{score_label} {metric}</span><span class="dataset">{dataset} &nbsp; ↗</span></button>')
        # The saved record's old exclude-6 diagnostic is not a verified no-data score.
        if task == 'm-chesapeake':
            section = re.sub(r'; mIoU excluding .*?: [0-9.]+', '; historical exclude-class diagnostic omitted because class meanings are unverified', section)
        inner = re.sub(r'^<section\b[^>]*><h2>.*?</h2>', '', section, count=1, flags=re.S)
        inner = re.sub(r'</section>$', '', inner)
        example_match = re.search(r'<h3>Fixed validation examples.*?(?=<h3>Compute</h3>|$)', inner, re.S)
        examples = example_match.group(0) if example_match else '<p class="muted">No saved prediction examples are available for this task.</p>'
        if example_match:
            inner = inner[:example_match.start()] + inner[example_match.end():]
            examples = re.sub(r'<h3>.*?</h3>', '<h3>See the predictions</h3><p class="muted">Fixed validation examples from the saved run. These are not a representative accuracy sample. For masks: original image → correct regions → model prediction. Colors denote class IDs.</p>', examples, count=1, flags=re.S)
        checks = 'Selected from full validation' if selected else 'Selection pending'
        limits = {
            'm-eurosat': 'Validation measures performance on this benchmark. It does not establish accuracy on imagery from a new region or sensor.',
            'm-pv4ger': 'Most images contain no panels. Check panel recall and per-class results in technical details; overall accuracy can hide missed panels.',
            'm-pv4ger-seg': 'Mean IoU averages panel and background overlap. Inspect panel-specific IoU and missed or merged panels in technical details.',
            'm-nz-cattle': 'Cattle occupy a small fraction of each image. Inspect cattle-only IoU and the marked regions; background accuracy alone can hide missed animals.',
            'm-chesapeake': 'The displayed mean includes all seven numeric labels. Class meanings remain unverified, and class 0 scores almost perfectly. Inspect each class before interpreting this as land-cover performance.',
        }[task]
        final = text(cells[7]) if cells else 'not evaluated'
        if test_match: final = 'Final test: ' + pct(test_match.group(1)) + ' ' + metric
        elif final == 'not evaluated': final = 'Final test: not evaluated'
        else: final = 'Final test evidence: ' + final
        panels.append(f'<section class="panel" id="panel-{task}" data-task-panel="{task}" {"hidden" if i else ""}><div class="panel-head"><div><span class="eyebrow">{dataset}</span><h2>{name}</h2><p class="muted">{purpose}</p></div><span class="tag">{checks}</span></div><p><b>{score}</b> validation {metric} · {esc(final)}</p><div class="limitation"><b>What to check before using this model</b><br>{limits}</div>{examples}<details class="technical"><summary>Explore experiments, learning curves and model details</summary>{inner}</details></section>')
    evidence = []
    for s in sections:
        title_match = re.search(r'<h2>(.*?)</h2>', s, re.S)
        title = text(title_match.group(1)) if title_match else 'Evidence'
        if any(re.match(r'<section\b[^>]*><h2>' + re.escape(t) + r'(?:\s|<)', s) for t in TASKS): continue
        content = re.sub(r'^<section\b[^>]*><h2>.*?</h2>', '', s, count=1, flags=re.S)
        content = re.sub(r'</section>$', '', content)
        evidence.append(f'<details><summary>{esc(title)}</summary>{content}</details>')
    header_match = re.search(r"<div class='sub'>(.*?)</div>", doc, re.S)
    generated = text(header_match.group(1)) if header_match else 'Generated from saved GoTrain records'
    banners = ''.join(re.findall(r"<div class='banner'>.*?</div>", doc, re.S))
    note = 'This is a saved report. Scores describe validation unless explicitly marked as final test results. Regenerate the report to include newer training or verification events.'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GoTrain | Model workspace</title><style>{CSS}</style></head><body>
<div class="shell"><aside class="rail"><div class="brand">Go<em>Train</em><span style="color:#116b56">.</span></div><small>From data to a selected model</small><nav aria-label="Report navigation"><a href="#overview">01 &nbsp; Model overview</a><a href="#task-detail">02 &nbsp; Predictions</a><a href="#evidence">03 &nbsp; Evidence &amp; logs</a><a href="#guide">04 &nbsp; Quick guide</a></nav></aside>
<main class="workspace"><div class="topline"><span class="eyebrow">Model workspace / saved report</span><button class="print" id="print-report">Print summary</button></div><div id="overview"><h1>Your models, explained.</h1><p class="intro">See what each model does, inspect its predictions, and understand the evidence behind its selection.</p></div>{banners}
<div class="notice"><b>Start with a task below.</b><p>Classification labels an image. Segmentation outlines regions inside it. Each task has its own score; scores across different tasks should not be ranked against each other.</p></div>
<div class="cards">{''.join(cards)}</div><div class="section-label" id="task-detail"><h2>Explore a selected model</h2><span>Choose a card above to switch tasks</span></div>{''.join(panels)}
<section class="evidence" id="evidence"><span class="eyebrow">Traceable decisions</span><h2>Evidence &amp; experiment history</h2><p class="muted">Original experiment records, agent decisions, storage checks and final-evaluation status. Missing evidence stays missing.</p>{''.join(evidence)}</section>
<section class="glossary" id="guide"><span class="eyebrow">Quick guide</span><h2>What the numbers mean</h2><div class="glossary-grid"><div><b>Accuracy</b>The fraction of images assigned the correct class. Class imbalance can make it look better than minority-class performance.</div><div><b>Mean IoU</b>Overlap between predicted and correct regions, averaged across classes. Check which classes are included.</div><div><b>Validation versus test</b>Validation guides model selection. The held-out test split is evaluated after selection is locked.</div><div><b>Stored versus verified</b>A stored model is a saved version. Verification checks file integrity and prediction reproduction on reference examples; it does not prove real-world accuracy.</div></div></section>
<footer>{esc(note)}<br><details><summary>Report source and generation time</summary>{esc(generated)}</details></footer></main></div><script>{JS}</script></body></html>'''
