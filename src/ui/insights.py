"""Escaped, responsive forecast explanations; no Streamlit or network dependency."""
from html import escape
import json
from pathlib import Path

INSIGHTS_CSS = """
<style>
.forecast-status{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:12px 0 18px;padding:12px 14px;background:linear-gradient(110deg,rgba(56,189,248,.08),rgba(198,255,58,.035));border:1px solid rgba(56,189,248,.18);border-radius:12px}
.forecast-status.seed{border-color:rgba(245,158,11,.3);background:rgba(245,158,11,.055)}
.forecast-status strong{color:#e5edf7;font-size:12px}.forecast-status small{display:block;color:#93a4b8;font-size:11px;line-height:1.6}
.source-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#38bdf8;margin-right:7px;box-shadow:0 0 10px #38bdf844}.seed .source-dot{background:#f59e0b}
.forecast-tag{font-size:10px;letter-spacing:.08em;color:#a4b5c8;border:1px solid #ffffff18;border-radius:6px;padding:4px 7px;white-space:nowrap}
.xi-insights{margin:16px 0 22px;border:1px solid #ffffff16;border-radius:14px;background:rgba(255,255,255,.025);overflow:hidden}
.xi-insights>summary{padding:15px 16px;cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:12px}.xi-insights>summary::-webkit-details-marker{display:none}
.insight-title{font-size:13px;font-weight:700;color:#dde8f5}.insight-sub{font-size:11px;color:#8b9aad;margin-top:4px;line-height:1.5}.insight-plus{font-size:19px;color:#38bdf8;font-weight:400}.xi-insights[open]>summary .insight-plus{transform:rotate(45deg)}
.insight-body{padding:0 16px 16px}.insight-calls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:10px 0 14px}
.insight-call{padding:12px;border:1px solid #ffffff10;border-radius:10px;background:#ffffff03;min-width:0}.insight-call .kicker{font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:#8a9aaf}.insight-call .names{font-size:12px;color:#dbe6f2;margin-top:7px;line-height:1.7;overflow-wrap:anywhere}.insight-call .edge{font-size:11px;color:#c6ff3a;margin-top:5px}.insight-call .muted{color:#8a9aaf}
.fpl p.insight-note{font-size:11px;color:#8b9aad;line-height:1.65;margin:8px 0}.insight-note.warn{color:#e6b974}
.projection-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-top:12px}
.projection-player{border:1px solid #ffffff10;border-radius:9px;min-width:0;background:#080e16}.projection-player>summary{padding:10px;cursor:pointer;display:flex;gap:8px;justify-content:space-between;font-size:11px;color:#cbd7e5;list-style:none}.projection-player>summary::-webkit-details-marker{display:none}
.projection-player .pts{color:#c6ff3a;font-variant-numeric:tabular-nums;white-space:nowrap}.projection-player .role{color:#7e91a9;font-size:9px;margin-right:5px}.projection-player .metrics{padding:0 10px 10px;display:grid;grid-template-columns:1fr 1fr;gap:10px}.projection-player .metrics span{display:block;color:#8295ad;font-size:9px;line-height:1.5}.projection-player .metrics b{color:#d9e5f1;font-size:12px;font-weight:600}.projection-player .risk{color:#e7b87c;font-size:10px;margin:0 10px 10px}.projection-player .fixture-badge{font-size:9px;color:#38bdf8;margin-left:5px}
.validation-metrics{display:flex;flex-wrap:wrap;gap:20px;margin:10px 0}.validation-metrics b{font-size:21px;color:#dbe9f6;display:block;font-weight:600}.validation-metrics span{font-size:10px;color:#8c9db2}.xi-insights summary:focus-visible{outline:2px solid #38bdf8;outline-offset:-3px}
@media(max-width:560px){.insight-calls,.projection-list{grid-template-columns:1fr}.insight-body{padding:0 12px 12px}.xi-insights>summary{padding:13px 12px}.forecast-status{padding:10px 12px}.forecast-tag{font-size:9px}}
</style>
"""


def text(value):
    return escape(str(value), quote=True)


def forecast_status(advice):
    projection = advice.get('projection', {})
    source = projection.get('source', 'season_model')
    seeded = source != 'season_model'
    labels = {'season_model': 'Season model', 'current_form_seed': 'Seeded from current form',
              'previous_season_seed': 'Seeded from last season'}
    cutoff = advice.get('form_gw')
    form = f'Form through GW{cutoff}' if cutoff else ('Previous-season form' if seeded else 'Historical form')
    reason = projection.get('reason')
    note = f"<small>{text(reason)}</small>" if reason else ''
    if advice.get('budget_estimated'):
        note += "<small>Selling prices partly estimated · check your FPL transfer budget.</small>"
    return (f"<div class='forecast-status{' seed' if seeded else ''}'><div><strong>"
            f"<i class='source-dot'></i>{text(labels.get(source, source))}</strong>"
            f"<small>Advice for GW{int(advice['gw'])} · {text(form)} · Live availability</small>{note}</div>"
            "<span class='forecast-tag'>EXPECTED POINTS</span></div>")


def projection_details(rows):
    players = []
    def metric(label, value):
        return f"<div><span>{label}</span><b>{value}</b></div>"
    for row in sorted(rows, key=lambda r: r['pred'], reverse=True):
        role = 'XI' if row['role'] == 'start' else 'BENCH'
        count = row.get('fixture_count')
        fixture_badge = 'DOUBLE' if count and count > 1 else ('BLANK' if count == 0 else '')
        metrics = []
        prefix = 'Seed' if row.get('signal_basis') == 'seed_average' else 'Recent'
        mins = row.get('recent_minutes')
        starts = row.get('recent_starts')
        p_play = row.get('p_play')
        form = row.get('recent_form')
        xgi = row.get('xgi')
        metrics.append(metric(f'{prefix} minutes / match', f'{mins:.0f}' if mins is not None else '—'))
        metrics.append(metric(f'{prefix} starts', f'{starts*100:.0f}%' if starts is not None else '—'))
        metrics.append(metric('Chance of minutes / match', f'{p_play*100:.0f}%' if p_play is not None else '—'))
        metrics.append(metric(f'{prefix} points / match', f'{form:.1f}' if form is not None else '—'))
        metrics.append(metric(f'{prefix} xGI / match', f'{xgi:.2f}' if xgi is not None else '—'))
        metrics.append(metric('Weighted next-4 avg', f"{row.get('pred_run', row['pred']):.1f}"))
        risk = "<div class='risk'>Watch minutes: recent average below 60.</div>" if mins is not None and mins < 60 else ''
        players.append(f"<details class='projection-player'><summary><span><span class='role'>{role}</span>"
                       f"{text(row['name'])}<span class='fixture-badge'>{fixture_badge}</span></span>"
                       f"<span class='pts'>{row['pred']:.1f} pts +</span></summary>"
                       f"<div class='metrics'>{''.join(metrics)}</div>{risk}</details>")
    return ''.join(players)


def lineup_insights(advice):
    calls = []
    for call in advice.get('selection', []):
        gap = call['gap']
        edge = f"+{gap:.2f} pts in the XI" if gap >= .005 else 'Effectively tied on projected points'
        calls.append("<div class='insight-call'><div class='kicker'>Legal lineup alternative</div>"
                     f"<div class='names'>{text(call['starter'])} <span class='muted'>over</span> {text(call['bench'])}</div>"
                     f"<div class='edge'>{edge}</div></div>")
    return ("<details class='xi-insights'><summary><div><div class='insight-title'>Your lineup, explained</div>"
            "<div class='insight-sub'>Compare the bench · inspect form and minutes</div></div><span class='insight-plus'>+</span></summary>"
            "<div class='insight-body'><p class='insight-note'>The XI maximises expected points under formation rules. "
            "Opposing players can both start. Comparisons below are legal one-player swaps, before captain bonus.</p>"
            f"<div class='insight-calls'>{''.join(calls)}</div><div class='insight-title'>Behind each projection</div>"
            "<p class='insight-note'>Tap a player for details. Recent minutes, starts and xGI describe observed form; "
            "seeded estimates use season-average form. These are not a calibrated prediction of starting. Chance of minutes is modelled per match. "
            "These are supporting signals, not a causal breakdown of the score.</p>"
            f"<div class='projection-list'>{projection_details(advice['current_xi'])}</div></div></details>")


def validation_panel(report_path=None):
    path = report_path or Path(__file__).resolve().parents[2] / 'docs' / 'model_validation.json'
    try:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        current = report['metrics']['current_blend']
        baseline = report['metrics']['form3']
    except (OSError, ValueError, KeyError):
        return ''
    second_note = ''
    if report_path is None:
        try:
            second = json.loads(path.with_name('model_validation_second.json').read_text(encoding='utf-8'))
            second_note = (f"<p class='insight-note'>Second holdout · {text(second['test_season'])}: "
                           f"model + form {second['metrics']['current_blend']['mae']:.2f} MAE · "
                           f"simple form {second['metrics']['form3']['mae']:.2f} MAE.</p>")
        except (OSError, ValueError, KeyError):
            pass
    return ("<details class='xi-insights'><summary><div><div class='insight-title'>Model check</div>"
            f"<div class='insight-sub'>Historical holdout · {text(report['test_season'])}</div></div><span class='insight-plus'>+</span></summary>"
            "<div class='insight-body'><div class='validation-metrics'>"
            f"<div><b>{current['mae']:.2f}</b><span>Model + form · MAE</span></div>"
            f"<div><b>{baseline['mae']:.2f}</b><span>Simple recent form · MAE</span></div></div>"
            f"<p class='insight-note'>Lower is better. Separate evaluation model trained on {text(report['train_season'])}, "
            f"tested on {text(report['test_season'])}: {int(report['test_player_gameweeks']):,} player-gameweeks. "
            "Only prior observed form is used. This does not measure the current live model or guarantee future accuracy.</p>"
            f"{second_note}<p class='insight-note'>Weight comparisons are exploratory. No tested alternative consistently "
            "improves every main error measure; production weights remain unchanged.</p></div></details>")
