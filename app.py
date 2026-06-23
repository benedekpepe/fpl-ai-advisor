"""FPL AI Advisor — Streamlit dashboard.

Run locally (with the Postgres container up and the model trained):
    streamlit run app.py
"""
import streamlit as st

from src.advisor.personal import build_advice

st.set_page_config(page_title="FPL AI Advisor", page_icon="⚽", layout="centered")

VERDICT = {
    "TRANSFER": ("#22c55e", "TRANSFER"),
    "HOLD": ("#38bdf8", "HOLD"),
    "BANK": ("#f59e0b", "HOLD / BANK"),
    "CHIP": ("#a855f7", "PLAY A CHIP"),
}

TEAM_ABBR = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Burnley": "BUR", "Chelsea": "CHE", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Ipswich": "IPS", "Leeds": "LEE",
    "Leicester": "LEI", "Liverpool": "LIV", "Luton": "LUT", "Man City": "MCI",
    "Manchester City": "MCI", "Man Utd": "MUN", "Manchester Utd": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Nottingham Forest": "NFO", "Sheffield Utd": "SHU",
    "Southampton": "SOU", "Spurs": "TOT", "Tottenham": "TOT", "Sunderland": "SUN",
    "West Ham": "WHU", "Wolves": "WOL",
}
CLUB_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#7A263A", "Bournemouth": "#DA291C", "Brentford": "#E30613",
    "Brighton": "#0057B8", "Burnley": "#6C1D45", "Chelsea": "#034694", "Crystal Palace": "#1B458F",
    "Everton": "#003399", "Fulham": "#1b1b1b", "Ipswich": "#3A64A3", "Leeds": "#FFCD00",
    "Leicester": "#003090", "Liverpool": "#C8102E", "Luton": "#F78F1E", "Man City": "#6CABDD",
    "Manchester City": "#6CABDD", "Man Utd": "#DA291C", "Manchester Utd": "#DA291C", "Newcastle": "#2b2a2a",
    "Nott'm Forest": "#DD0000", "Nottingham Forest": "#DD0000", "Sheffield Utd": "#EE2737",
    "Southampton": "#D71920", "Spurs": "#132257", "Tottenham": "#132257", "Sunderland": "#EB172B",
    "West Ham": "#7A263A", "Wolves": "#FDB913",
}


def team_code(team):
    return "" if not team else TEAM_ABBR.get(team, str(team)[:3].upper())


def club_color(team):
    return CLUB_COLORS.get(team, "#5b6b7b")


AUTHOR_URL = "https://benedekpeter.netlify.app/"
CREDIT = (f"<a class='credit' href='{AUTHOR_URL}' target='_blank' rel='noopener'>"
          f"by B.P. Studio · Péter Benedek ↗︎</a>")


# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap');
.fpl{font-family:'DM Sans',sans-serif;color:#e8edf2;line-height:1.4}
.fpl *{box-sizing:border-box;cursor:default}
.fpl-wrap{background:radial-gradient(1100px 520px at 50% -12%,#15202c 0%,#0a0e14 62%);
  border:1px solid rgba(255,255,255,.08);border-radius:22px;padding:24px 22px 18px;
  box-shadow:0 30px 80px rgba(0,0,0,.45)}
.fpl h1,.fpl h2,.fpl h3{font-family:'Bricolage Grotesque',sans-serif;letter-spacing:-.02em;margin:0}
.wordmark{font-family:'Bricolage Grotesque';font-weight:800;font-size:26px;letter-spacing:-.03em;line-height:1.35;padding-top:1px}
.rhead{font-family:'Bricolage Grotesque';font-weight:700;font-size:18px;letter-spacing:-.01em;line-height:1.3;margin-bottom:8px}
.wordmark .dot{color:#c6ff3a}
.sub{color:#8a97a6;font-size:13px;margin-top:2px}
.head{display:flex;justify-content:space-between;align-items:center;gap:12px}
.credit{font-size:12px;color:#8a97a6;white-space:nowrap;margin-top:6px}
.fpl a.credit{color:#8a97a6;text-decoration:none}
.fpl a.credit:hover{color:#38bdf8}
.fpl a{color:#38bdf8;text-decoration:none;cursor:pointer}
.fpl a:hover{text-decoration:underline}
.lbl{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#7d8a99;font-weight:700;margin:24px 0 10px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);border-radius:999px;
  padding:6px 13px;font-size:12.5px;color:#cdd6e0}
.chip b{color:#fff;font-weight:700}
.hero{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:10px;margin-top:16px}
.cell{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:13px 14px}
.cell .k{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:#7d8a99;font-weight:700}
.cell .v{font-size:15px;font-weight:700;margin-top:4px;color:#fff}
.cell.lead{border-color:rgba(198,255,58,.4);box-shadow:inset 0 0 0 1px rgba(198,255,58,.12)}
.pitch{position:relative;border-radius:16px;padding:14px 6px;
  background:repeating-linear-gradient(0deg,#1d7a3e 0 36px,#1a6e39 36px 72px);
  border:2px solid rgba(255,255,255,.22)}
.pitch:before{content:'';position:absolute;inset:7px;border:2px solid rgba(255,255,255,.20);border-radius:9px;pointer-events:none}
.pitch:after{content:'';position:absolute;left:50%;top:50%;width:74px;height:74px;
  transform:translate(-50%,-50%);border:2px solid rgba(255,255,255,.16);border-radius:50%;pointer-events:none}
.prow{display:flex;justify-content:space-around;align-items:flex-start;gap:4px;position:relative;z-index:1;margin:8px 2px}
.pl{flex:1 1 0;min-width:0;max-width:98px;text-align:center;animation:rise .5s ease both}
.shirt{position:relative;background:rgba(8,12,18,.82);border:1px solid rgba(255,255,255,.14);border-radius:12px;
  padding:7px 5px 6px;box-shadow:0 6px 16px rgba(0,0,0,.4)}
.pl-name{font-size:11.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pl-meta{display:flex;align-items:center;justify-content:center;gap:6px;margin-top:2px}
.pl-pred{font-size:13px;font-weight:800;color:#c6ff3a}
.tm{font-size:9.5px;font-weight:700;letter-spacing:.06em;color:#9fb0c0;background:rgba(255,255,255,.08);
  border-radius:5px;padding:1px 5px}
.fx{display:flex;flex-wrap:wrap;justify-content:center;gap:3px;margin-top:5px}
.fxc{font-size:8.5px;font-weight:800;color:#fff;border-radius:4px;padding:1px 4px;min-width:24px;text-align:center;opacity:.95}
.d1{background:#0e7a3a}.d2{background:#3aa15e}.d3{background:#5f6b78}.d4{background:#d9802f}.d5{background:#cf3b3b}
.fxleg{display:flex;align-items:center;flex-wrap:wrap;gap:12px;color:#7d8a99;font-size:11px;margin-top:9px}
.fxleg .scale{display:inline-flex;align-items:center;gap:5px}
.fxleg .sw{width:15px;height:10px;border-radius:3px;display:inline-block}
.bench{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:11px;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:9px 12px}
.bench .bl{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:#7d8a99;font-weight:700}
.bench .bp{position:relative;font-size:12.5px;color:#dde5ee;background:rgba(255,255,255,.05);border-radius:8px;padding:3px 9px}
.bench.bb{border-color:rgba(198,255,58,.4);background:rgba(198,255,58,.07)}
.bench.bb .bl{color:#c6ff3a}
.bench.bb .bp{color:#e8ffc0;background:rgba(198,255,58,.14)}
.bdg{display:inline-block;min-width:16px;height:16px;line-height:15px;border-radius:50%;
  font-size:9.5px;font-weight:800;color:#0a0e14;background:#ffd54a;margin-left:4px;vertical-align:middle}
.bdg.v{background:#cdd8e6}
.caps{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:4px}
.capc{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:11px 12px}
.capc.lead{border-color:rgba(255,213,74,.5);box-shadow:inset 0 0 0 1px rgba(255,213,74,.15)}
.capc .rk{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#7d8a99;font-weight:700}
.capc .nm{font-weight:700;font-size:13.5px;margin-top:3px}
.capc .dt{font-size:12px;color:#aeb9c6;margin-top:3px}
.capc .pct{color:#ffd54a;font-weight:700}
.legend{color:#7d8a99;font-size:11.5px;margin:10px 0 4px}
.badge{display:inline-block;padding:4px 12px;border-radius:8px;font-weight:800;font-size:12px;letter-spacing:.05em;color:#0a0e14}
.mv{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:14px}
.mv .o{color:#ff8585;font-weight:600}.mv .i{color:#7ef0a6;font-weight:600}.mv .ar{color:#7d8a99}
.mvp{color:#8a97a6;font-weight:500;font-size:12px;margin-left:4px}
.nbtag{font-size:9.5px;font-weight:700;color:#9fb0c0;background:rgba(255,255,255,.07);border-radius:6px;padding:2px 7px}
.nbtag.worth{color:#c6ff3a;background:rgba(198,255,58,.12)}
.note{color:#8a97a6;font-size:12.5px;margin-top:6px}
.callout{display:flex;gap:9px;align-items:flex-start;background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08);border-left:3px solid #8a97a6;border-radius:8px;
  padding:9px 12px;font-size:12.5px;color:#cdd6e0;margin-top:10px}
.callout.good{border-left-color:#22c55e}
.callout .ic{font-weight:800;color:#8a97a6}
.callout.good .ic{color:#22c55e}
.crow{display:flex;flex-direction:column;gap:6px;margin:10px 0;padding:9px 11px;border-radius:11px;border:1px solid transparent}
.crow.rec{background:rgba(198,255,58,.07);border-color:rgba(198,255,58,.35)}
.crow.used{opacity:.5}
.crow .top{display:flex;align-items:center;justify-content:space-between;gap:10px}
.cname{font-weight:700;font-size:14px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.cval{font-size:12px;color:#aeb9c6;white-space:nowrap}
.cbar{height:7px;border-radius:6px;background:rgba(255,255,255,.08);overflow:hidden}
.cbar>i{display:block;height:100%;background:linear-gradient(90deg,#5eead4,#c6ff3a)}
.cbest{font-size:11.5px;color:#8b97a4}
.recpill{font-size:9.5px;font-weight:800;color:#0a0e14;background:#c6ff3a;border-radius:6px;padding:2px 7px}
.foot{color:#5f6b78;font-size:11px;margin-top:18px;border-top:1px solid rgba(255,255,255,.07);
  padding-top:12px;display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}
.foot span{max-width:560px}
.fpl a.foot-link{color:#7d8a99}
.tip:hover{z-index:20}
.tip:hover::after{content:attr(data-tip);position:absolute;left:50%;bottom:calc(100% + 8px);transform:translateX(-50%);
  background:#0b1118;color:#e8edf2;border:1px solid rgba(255,255,255,.16);border-radius:8px;padding:5px 9px;
  font-size:11px;line-height:1.2;white-space:nowrap;box-shadow:0 12px 26px rgba(0,0,0,.6);z-index:30;pointer-events:none}
.tip:hover::before{content:'';position:absolute;left:50%;bottom:calc(100% + 3px);transform:translateX(-50%);
  border:5px solid transparent;border-top-color:#0b1118;z-index:30;pointer-events:none}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@media (max-width:560px){
  .head{gap:8px}
  .wordmark{font-size:19px}
  .credit{font-size:10px}
  .foot{flex-direction:column;gap:8px}
  .foot-link{align-self:flex-end}
  .hero{grid-template-columns:1fr}
  .caps{grid-template-columns:1fr}
  .pl-name{font-size:10px}
  .pl-pred{font-size:12px}
  .fxc{min-width:18px;font-size:7.5px;padding:1px 3px}
  .fpl-wrap{padding:18px 13px 14px}
}
</style>
"""

APP_CSS = """
<style>
.stApp{background:#070a0f}
div.block-container{max-width:880px;padding-top:2.6rem}
.stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp h1, .stApp h2{cursor:default}
.stApp input[type="text"]{cursor:text}
div[data-baseweb="slider"], div[data-baseweb="slider"] *,
div[data-baseweb="slider"] [role="slider"]:active{cursor:pointer !important}
.stButton>button,.stButton>button *,.stFormSubmitButton>button,.stFormSubmitButton>button *{cursor:pointer !important}
.stButton>button,.stFormSubmitButton>button{background:#38bdf8;color:#06121c;border:0;
  border-radius:10px;font-weight:800;padding:.5rem 1.1rem}
.stButton>button:hover,.stFormSubmitButton>button:hover{background:#5cc8fa;color:#06121c}
@media (max-width:560px){div.block-container{padding-left:.5rem;padding-right:.5rem;padding-top:3.2rem}}
</style>
"""


# ---------------------------------------------------------------- helpers
def short_name(n):
    parts = (n or "").split()
    if not parts:
        return n
    if len(parts) == 1:
        return parts[0][:13]
    return f"{parts[0][0]}. {parts[-1]}"[:15]


def _tip(r):
    pr = f" · £{r['price']/10:.1f}m" if r.get("price") else ""
    return f"{r['name']} · {team_code(r.get('team'))}{pr}".replace('"', "")


def season_fmt(s):
    if s and "-" in s:
        a, b = s.split("-")
        if len(b) == 2:
            b = a[:2] + b
        return f"{a}/{b}"
    return s


def xi_summary(rows):
    starters = [r for r in rows if r["role"] == "start"]
    nd = sum(r["position"] == "DEF" for r in starters)
    nm = sum(r["position"] == "MID" for r in starters)
    nf = sum(r["position"] == "FWD" for r in starters)
    total = sum(r["pred"] for r in starters) + next((r["pred"] for r in starters if r["captain"]), 0.0)
    return f"{nd}-{nm}-{nf}", total


def _badge(r):
    if r["captain"]:
        return "<span class='bdg'>C</span>"
    if r.get("vice"):
        return "<span class='bdg v'>V</span>"
    return ""


def _fix(r):
    fx = r.get("fix") or []
    if not fx:
        return ""
    cells = "".join(
        f"<span class='fxc d{f['diff']}'>{f['opp']}</span>" for f in fx)
    return f"<div class='fx'>{cells}</div>"


def _pl(r):
    col = club_color(r.get("team"))
    return (f"<div class='pl'><div class='shirt tip' data-tip=\"{_tip(r)}\" style='border-top:3px solid {col}'>"
            f"<div class='pl-name'>{short_name(r['name'])}{_badge(r)}</div>"
            f"<div class='pl-meta'><span class='tm'>{team_code(r.get('team'))}</span>"
            f"<span class='pl-pred'>{r['pred']:.1f}</span></div>{_fix(r)}</div></div>")


def pitch_html(rows, bb=False):
    start = [r for r in rows if r["role"] == "start"]
    out = ["<div class='pitch'>"]
    for posn in ("FWD", "MID", "DEF", "GK"):
        line = [r for r in start if r["position"] == posn]
        if line:
            out.append("<div class='prow'>" + "".join(_pl(r) for r in line) + "</div>")
    out.append("</div>")
    gk = [r for r in rows if r["role"] == "benchgk"]
    bench = [r for r in rows if r["role"] == "bench"]
    chips = "".join(
        f"<span class='bp tip' data-tip=\"{_tip(r)}\">{short_name(r['name'])} "
        f"<span class='tm'>{team_code(r.get('team'))}</span> {r['pred']:.1f}</span>"
        for r in (gk + bench))
    bcls = " bb" if bb else ""
    blabel = "Bench Boost" if bb else "Bench"
    out.append(f"<div class='bench{bcls}'><span class='bl'>{blabel}</span>{chips}</div>")
    return "".join(out)


FX_LEGEND = ("<div class='fxleg'><span>Next 4 fixtures</span>"
             "<span class='scale'>easy<i class='sw d1'></i><i class='sw d2'></i>"
             "<i class='sw d3'></i><i class='sw d4'></i><i class='sw d5'></i>hard</span></div>")


def summary_caption(rows):
    form, total = xi_summary(rows)
    cap = next((r["name"] for r in rows if r.get("captain")), "")
    return (f"<div class='note' style='margin-top:8px'>Formation <b>{form}</b> · "
            f"Projected <b style='color:#c6ff3a'>{total:.1f} pts</b> (incl. captain) · "
            f"Captain <b>{cap}</b></div>")


def money_line(bank_after, ft_left, preserved=False):
    word = "preserved" if preserved else "left"
    return ("<div class='meta' style='margin:10px 0 14px'>"
            f"<span class='chip'>Bank after <b>£{bank_after/10:.1f}m</b></span>"
            f"<span class='chip'>Free transfers {word} <b>{ft_left}</b></span></div>")


def render_html(a):
    s = a["summary"]
    vc, vlabel = VERDICT[a["transfer"]["verdict"]]
    cap_v = short_name(s["captain"]) + ("  ×3" if s["triple"] else "")
    bb = (a["chips"]["recommended"] == "bboost")    # highlight bench when Bench Boost is the call

    tv = a["transfer"]["verdict"]
    if tv == "TRANSFER":
        htr = "".join(
            "<div style='white-space:nowrap'>"
            f"<span style='color:#ff8585'>{short_name(m['out'])}</span>"
            "<span style='color:#7d8a99'> → </span>"
            f"<span style='color:#7ef0a6'>{short_name(m['in'])}</span></div>"
            for m in a["transfer"]["moves"])
    elif tv == "HOLD":
        htr = "Hold"
    elif tv == "BANK":
        htr = "Bank FT"
    else:
        htr = "Played via chip"

    meta = (f"<div class='meta'>"
            f"<span class='chip'>Free transfers <b>{a['ft']}</b></span>"
            f"<span class='chip'>Bank <b>£{a['bank']/10:.1f}m</b></span>"
            f"<span class='chip'>Chip deadline <b>GW{a['deadline_gw']}</b></span></div>")

    chip_col = "#a855f7" if a["chips"]["recommended"] else "#8a97a6"
    hero = (f"<div class='hero'>"
            f"<div class='cell lead'><div class='k'>Transfers</div>"
            f"<div class='v' style='color:{vc}'>{htr}</div></div>"
            f"<div class='cell'><div class='k'>Captain</div>"
            f"<div class='v' style='color:#38bdf8'>{cap_v}</div></div>"
            f"<div class='cell'><div class='k'>Chip</div>"
            f"<div class='v' style='color:{chip_col}'>{s['chip']}</div></div></div>")

    # captain picks
    opts = a.get("captain_options", [])
    capblock = ""
    if opts:
        ranks = ["Captain", "Vice", "Option"]
        cards = []
        for i, o in enumerate(opts):
            if i == 0:
                pc = "#22c55e" if o["share"] >= 38 else "#f59e0b"
            else:
                pc = "#9aa6b2"
            cards.append(
                f"<div class='capc{' lead' if i == 0 else ''}'>"
                f"<div class='rk'>{i+1} · {ranks[i] if i < len(ranks) else 'Option'}</div>"
                f"<div class='nm'>{short_name(o['name'])} <span class='tm'>{team_code(o.get('team'))}</span></div>"
                f"<div class='dt'>{o['pred']:.1f} pts · <span class='pct' style='color:{pc}'>{o['share']}%</span></div></div>")
        capblock = ("<div class='lbl'>Captain picks</div>"
                    f"<div class='caps'>{''.join(cards)}</div>"
                    "<div class='legend'>Green = clear captain pick · amber = close call. "
                    "% is each pick's share of the trio's projected points.</div>")

    # transfers
    t = a["transfer"]
    tblock = ["<div class='lbl'>Transfer plan</div>",
              f"<span class='badge' style='background:{vc}'>{vlabel}</span> "
              f"<span style='font-size:13.5px;color:#cdd6e0'>{t['text']}</span>"]
    if t.get("moves"):
        if t["verdict"] == "CHIP":
            tblock.append("<div class='note'>Suggested squad for the chip "
                          "(vs your current team):</div>")
        for m in t["moves"]:
            op = f"<span class='mvp'>£{m['out_price']/10:.1f}</span>" if m.get("out_price") else ""
            ip = f"<span class='mvp'>£{m['in_price']/10:.1f}</span>" if m.get("in_price") else ""
            tblock.append(f"<div class='mv'><span class='o'>OUT {short_name(m['out'])}{op}</span>"
                          f"<span class='ar'>→</span><span class='i'>IN {short_name(m['in'])}{ip}</span></div>")
    if t["verdict"] == "TRANSFER":
        tblock.append(f"<div class='note'>{t['k']} transfer(s), {t['hits']} hit(s), -{4*t['hits']} pts</div>")
        tblock.append("<div class='lbl' style='margin-top:14px'>Your team after the move(s)</div>")
        tblock.append(money_line(t["bank_after"], t["ft_left"]))
        tblock.append(pitch_html(t["post_xi"], bb=bb) + FX_LEGEND + summary_caption(t["post_xi"]))
    if t.get("hit_note"):
        good = " good" if t.get("hit_worth") else ""
        icon = "✓" if t.get("hit_worth") else "ℹ"
        tblock.append(f"<div class='callout{good}'><span class='ic'>{icon}</span>"
                      f"<span>{t['hit_note']}</span></div>")

    # chips
    c = a["chips"]
    cblock = [f"<div class='lbl'>Chips · {a['half_label']} · one per gameweek</div>",
              "<div class='legend' style='margin-top:-2px'>Bar = how close this gameweek is to the chip's best "
              "week — a full bar means now is a great time to play it. ‘This GW' is its value if played now.</div>"]
    if not c["table"]:
        cblock.append("<div class='note'>No chips this half.</div>")
    else:
        for r in c["table"]:
            rec = (r["chip"] == c["recommended"])
            if r.get("used"):
                cblock.append(f"<div class='crow used'><div class='top'><div class='cname'>{r['label']}</div>"
                              f"<div class='cval'>used GW{r['used_gw']}</div></div>"
                              f"<div class='cbest'>already played this half</div></div>")
                continue
            if r.get("no_data"):
                cblock.append(f"<div class='crow'><div class='top'><div class='cname'>{r['label']}</div>"
                              f"<div class='cval'>not enough data</div></div></div>")
                continue
            second = (f" · then GW{r['second_gw']} (+{r['second_val']:.1f})" if "second_gw" in r else "")
            tv_num = r["this_val"]
            this = "n/a" if tv_num is None else f"+{tv_num:.1f}"
            fill = int(max(0.0, min(1.0, (tv_num / r["best_val"])
                                    if (tv_num is not None and r["best_val"] > 0) else 0.0)) * 100)
            pill = "<span class='recpill'>PLAY NOW</span>" if rec else ""
            nb = ""
            if not rec:
                if r.get("near_best"):
                    nb = "<span class='nbtag worth'>worth it now</span>"
                elif r["best_gw"] > a["gw"]:
                    nb = f"<span class='nbtag'>peaks GW{r['best_gw']}</span>"
            cblock.append(
                f"<div class='crow{' rec' if rec else ''}'>"
                f"<div class='top'><div class='cname'>{r['label']}{pill}{nb}</div>"
                f"<div class='cval'>this GW {this}</div></div>"
                f"<div class='cbar'><i style='width:{fill}%'></i></div>"
                f"<div class='cbest'>best GW{r['best_gw']} (+{r['best_val']:.1f}){second}</div></div>")
        rec = c["recommended"]
        if rec:
            d = c["detail"]
            if d["type"] in ("wildcard", "freehit"):
                kind = "Wildcard — keep going forward" if d["type"] == "wildcard" else "Free Hit — one week only"
                cblock.append(f"<div class='lbl' style='margin-top:14px'>New squad · {kind}</div>")
                cblock.append(money_line(d["bank_after"], d["ft_left"], preserved=True))
                cblock.append(pitch_html(d["squad"]) + FX_LEGEND + summary_caption(d["squad"]))
            elif d["type"] == "bboost":
                cblock.append(f"<div class='note'>Bench Boost: your bench (highlighted green above) "
                              f"adds a projected <b style='color:#c6ff3a'>{d['bench_pts']:.1f} pts</b> this week.</div>")
            elif d["type"] == "3xc":
                cblock.append(f"<div class='note'>Triple Captain on <b>{d['player']}</b> "
                              f"(pred {d['pred']:.2f} → ×3).</div>")
        elif a["half"] == 1:
            cblock.append(f"<div class='note'>Hold — every remaining chip's best week is still "
                          f"ahead (play them before the GW{a['deadline_gw']} deadline).</div>")
        else:
            cblock.append("<div class='note'>Hold — every remaining chip's best week is still ahead.</div>")

    return (CSS + "<div class='fpl'><div class='fpl-wrap'>"
            f"<div class='rhead'>Gameweek {a['gw']} · Team {a['team_id']} · {season_fmt(a['season'])}</div>"
            f"{meta}{hero}"
            f"<div class='lbl'>Best XI from your current squad</div>{pitch_html(a['current_xi'], bb=bb)}{FX_LEGEND}"
            f"{summary_caption(a['current_xi'])}"
            + capblock + "".join(tblock) + "".join(cblock) +
            "<div class='foot'><span>Unofficial tool — not affiliated with the Premier League or FPL. "
            "Predictions are uncertain; use as a guide, not gospel.</span>"
            f"<a class='foot-link' href='{AUTHOR_URL}' target='_blank' rel='noopener'>"
            "Péter Benedek · B.P. Studio</a></div>"
            "</div></div>")


# ---------------------------------------------------------------- gate
def gate():
    try:
        configured = st.secrets.get("app_password")
    except Exception:
        configured = None
    if not configured:
        return True
    if st.session_state.get("authed"):
        return True
    pw = st.text_input("Access password", type="password")
    if pw and pw == configured:
        st.session_state["authed"] = True
        st.rerun()          # re-run immediately so the prompt disappears
    if pw:
        st.error("Wrong password.")
    return False


# ---------------------------------------------------------------- page
def main():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown("<div class='fpl'>"
                "<div class='head'>"
                "<div class='wordmark'>FPL <span class='dot'>·</span> AI Advisor</div>"
                f"{CREDIT}</div>"
                "</div>" + CSS, unsafe_allow_html=True)

    if not gate():
        return

    with st.form("inputs"):
        st.markdown("<div style='color:#8a97a6;font-size:13px;margin-bottom:4px'>Enter your FPL Team ID "
                    "and a gameweek for a data-driven weekly plan.</div>", unsafe_allow_html=True)
        team_id = st.text_input("FPL Team ID", value="", placeholder="e.g. 1234567",
                                help="The number in your FPL team URL: /entry/<this>/")
        gw = st.select_slider("Gameweek", options=list(range(1, 39)), value=6)
        st.caption("Next season the gameweek will auto-detect; for now pick one.")
        submitted = st.form_submit_button("Get advice")

    if submitted:
        if not team_id.strip():
            st.info("Enter your FPL Team ID above to get advice.")
            return
        if not team_id.strip().isdigit():
            st.error("Team id must be a number (find it in your FPL team URL).")
            return
        with st.spinner("Crunching the season's numbers and optimising — the first run "
                        "takes up to a minute; after that it's quick…"):
            try:
                advice = build_advice(int(team_id), int(gw))
                if not advice["ok"]:
                    st.warning(advice["error"])
                else:
                    st.markdown(render_html(advice), unsafe_allow_html=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                st.caption("Is the Postgres container running and the model trained? "
                           "The advisor needs both, just like the CLI.")


if __name__ == "__main__":
    main()
