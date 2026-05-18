"""
NQ ICT Strategy v3  –  Backtest3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategie: FVG/IFVG + Liquidity Sweep + Kill Zones + Fibonacci OTE ...

  1. Compound Sizing   : 0.5 % des aktuellen Equity-Standes (Basis)
  2. Drawdown-Bremsen :
       DD > 3.5 % → 0.25 % Risiko   (halbe Größe)
       DD > 5.5 % → 0.125 % Risiko  (viertel Größe)
  3. Safety-Cap       : vor jedem Entry wird das Risiko so begrenzt,
                        dass selbst ein Volltreffer-SL das 7%-Limit
                        nicht überschreitet (mathematische Garantie)
  4. Kein Reset       : peak_eq = All-Time-High, wird nie zurückgesetzt
                        → Strategie handelt die gesamten 5 Jahre
  5. Tages-SL-Bremse  : max. 2 aufeinanderfolgende SLs pro Tag
  6. Max. 3 Trades/Tag
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
DATA_DIR   = Path(__file__).parent / "data"
OUT_DIR    = Path(__file__).parent / "results"
START_DATE = "2021-05-15"
END_DATE   = "2026-05-15"

INIT_EQ = 100_000.0
NQ_PV   = 20.0

BASE_RISK_PCT   = 0.005   # 0.5 % des aktuellen Equity
DD_BRAKE_SOFT   = 0.035   # 3.5 % DD → 0.25 % Risiko
DD_BRAKE_HARD   = 0.055   # 5.5 % DD → 0.125 % Risiko
DD_MAX          = 0.070   # 7.0 % DD → Safety-Cap greift, kein Handel über Limit
MAX_CONSEC_SL   = 2       # max. aufeinanderfolgender SLs / Tag

SL_BUF  = 5.0
MAX_SL  = 20.0
MIN_RR  = 2.0

SWING_N_1H  = 5
SWING_N_15M = 6
SWING_N_5M  = 8

FVG_MIN      = 4.0
ZONE_MAX_AGE = 20
EQL_LOOKBACK = 20

SWEEP_VALID_BARS = 30
SWEEP_LOOKBACK   = 10

KZ1_START, KZ1_END = 8.5,  11.0
KZ2_START, KZ2_END = 13.5, 15.0

MAX_TRADES_DAY = 3

MC_SIMS  = 200
BLOCK_SZ = 8
PERM_N   = 5000

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# DYNAMISCHES RISIKO basierend auf aktuellem DD
# ═══════════════════════════════════════════════════════════════
def effective_risk(current_eq: float, peak_eq: float) -> float:
    dd = (peak_eq - current_eq) / peak_eq
    if dd >= DD_BRAKE_HARD: return BASE_RISK_PCT * 0.25   # 0.125 %
    if dd >= DD_BRAKE_SOFT: return BASE_RISK_PCT * 0.50   # 0.250 %
    return BASE_RISK_PCT                                   # 0.5 %

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_nq(name: str) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / name, sep=';', header=None,
        names=['date','time','open','high','low','close','volume']
    )
    df['datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S'
    )
    df = (df.set_index('datetime')
            .drop(columns=['date','time'])
            .astype(float).sort_index()
            .loc[START_DATE:END_DATE])
    return df[~df.index.duplicated(keep='first')]

# ═══════════════════════════════════════════════════════════════
# ICT KONZEPT 1 – SWING PIVOTS  (n-Bar-Bestätigung, kein Look-Ahead)
# ═══════════════════════════════════════════════════════════════
def detect_pivots(df: pd.DataFrame, n: int) -> pd.DataFrame:
    hi, lo = df['high'], df['low']
    rh = hi.rolling(2*n+1, center=True, min_periods=2*n+1).max()
    rl = lo.rolling(2*n+1, center=True, min_periods=2*n+1).min()
    is_sh = (hi == rh) & (hi > hi.shift(1)) & (hi > hi.shift(-1))
    is_sl = (lo == rl) & (lo < lo.shift(1))  & (lo < lo.shift(-1))
    sh_c = is_sh.shift(n).fillna(False).astype(bool)
    sl_c = is_sl.shift(n).fillna(False).astype(bool)
    return pd.DataFrame({
        'sh': sh_c, 'sl': sl_c,
        'sh_val': hi.shift(n).where(sh_c),
        'sl_val': lo.shift(n).where(sl_c),
    }, index=df.index)

# ═══════════════════════════════════════════════════════════════
# ICT KONZEPT 2 – MARKTSTRUKTUR  (BOS / CHoCH / MSS)
# ═══════════════════════════════════════════════════════════════
def compute_bias(pivots: pd.DataFrame) -> pd.Series:
    bias = pd.Series('neutral', index=pivots.index, dtype=object)
    last_sh, last_sl, cur = [], [], 'neutral'
    for t, row in pivots.iterrows():
        if row['sh'] and pd.notna(row['sh_val']):
            last_sh.append(row['sh_val'])
            if len(last_sh) > 2: last_sh.pop(0)
        if row['sl'] and pd.notna(row['sl_val']):
            last_sl.append(row['sl_val'])
            if len(last_sl) > 2: last_sl.pop(0)
        if len(last_sh) == 2 and len(last_sl) == 2:
            hh = last_sh[1] > last_sh[0]; hl = last_sl[1] > last_sl[0]
            lh = last_sh[1] < last_sh[0]; ll = last_sl[1] < last_sl[0]
            if hh or hl:   cur = 'bull'
            elif lh or ll: cur = 'bear'
        bias[t] = cur
    return bias

def _to_arr(s: pd.Series) -> np.ndarray:
    a = np.zeros(len(s), dtype=np.int8)
    a[s.values == 'bull'] =  1
    a[s.values == 'bear'] = -1
    return a

# ═══════════════════════════════════════════════════════════════
# ICT KONZEPT 3 – FVG  (IFVG inline in Simulation)
# ═══════════════════════════════════════════════════════════════
def precompute_fvg(df: pd.DataFrame) -> list:
    h, l, ts = df['high'].values, df['low'].values, df.index
    zones = []
    for i in range(len(df) - 2):
        gb = l[i+2] - h[i]
        if gb >= FVG_MIN:
            zones.append(dict(type='FVG', dir='bull',
                top=l[i+2], bottom=h[i], entry=l[i+2], sl=h[i]-SL_BUF,
                origin_bar=i+2, origin_ts=ts[i+2], active=True, used=False))
        gr = l[i] - h[i+2]
        if gr >= FVG_MIN:
            zones.append(dict(type='FVG', dir='bear',
                top=l[i], bottom=h[i+2], entry=h[i+2], sl=l[i]+SL_BUF,
                origin_bar=i+2, origin_ts=ts[i+2], active=True, used=False))
    return zones

# ═══════════════════════════════════════════════════════════════
# EQUAL HIGHS / LOWS
# ═══════════════════════════════════════════════════════════════
def recent_eql(sh: list, sl: list, tol: float = 4.0):
    eqh, eql = [], []
    for i in range(len(sh)):
        for j in range(i+1, len(sh)):
            if abs(sh[i]-sh[j]) <= tol: eqh.append((sh[i]+sh[j])/2)
    for i in range(len(sl)):
        for j in range(i+1, len(sl)):
            if abs(sl[i]-sl[j]) <= tol: eql.append((sl[i]+sl[j])/2)
    return np.array(eqh), np.array(eql)

# ═══════════════════════════════════════════════════════════════
# METRIKEN
# ═══════════════════════════════════════════════════════════════
def calc_metrics(trades: pd.DataFrame, init_eq: float = INIT_EQ) -> dict:
    if len(trades) == 0: return {}
    t = trades.copy()
    t['exit_ts']  = pd.to_datetime(t['exit_ts'])
    t['entry_ts'] = pd.to_datetime(t['entry_ts'])
    t = t.sort_values('exit_ts')

    pnl   = t['pnl_usd'].values
    curve = np.concatenate([[init_eq], init_eq + np.cumsum(pnl)])

    wins  = pnl[pnl > 0]; losses = pnl[pnl < 0]
    wr    = len(wins) / len(pnl)
    pf    = wins.sum() / max(-losses.sum(), 1e-9)
    avg_w = wins.mean()  if len(wins)   else 0.0
    avg_l = losses.mean() if len(losses) else 0.0
    exp   = wr*avg_w + (1-wr)*avg_l

    r_mult     = t['pnl_pts'] / t['sl_dist']
    avg_r_win  = r_mult[r_mult>0].mean() if (r_mult>0).any() else 0.0
    avg_r_loss = r_mult[r_mult<0].mean() if (r_mult<0).any() else 0.0
    win_pts    = t.loc[t['pnl_pts']>0,'pnl_pts'].mean() if (t['pnl_pts']>0).any() else 0.0
    loss_pts   = t.loc[t['pnl_pts']<0,'pnl_pts'].mean() if (t['pnl_pts']<0).any() else 0.0
    hold_min   = (t['exit_ts']-t['entry_ts']).dt.total_seconds().mean()/60

    peak   = np.maximum.accumulate(curve)
    dd     = (curve - peak) / peak
    max_dd = dd.min()

    t['day'] = t['exit_ts'].dt.date
    daily_pnl = t.groupby('day')['pnl_usd'].sum()
    all_days  = pd.date_range(START_DATE, END_DATE, freq='B')
    daily_pnl = daily_pnl.reindex(all_days.date, fill_value=0.0)
    dr     = daily_pnl / init_eq
    sharpe = dr.mean() / max(dr.std(), 1e-9) * np.sqrt(252)

    # Always use the full backtest window so a hard-stop doesn't inflate CAGR
    years  = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 365.25
    cagr   = (curve[-1]/curve[0]) ** (1/max(years,0.1)) - 1
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    by_zone = {}
    for zt, grp in t.groupby('zone_type'):
        rm = grp['pnl_pts'] / grp['sl_dist']
        by_zone[zt] = dict(
            n=len(grp), wr=f"{(grp['pnl_usd']>0).mean():.1%}",
            pnl=f"${grp['pnl_usd'].sum():,.0f}", avg_r=f"{rm.mean():+.3f}R")

    return dict(
        n_trades=len(t), win_rate=wr, loss_rate=1-wr,
        profit_factor=pf, avg_win_usd=avg_w, avg_loss_usd=avg_l,
        avg_win_pts=win_pts, avg_loss_pts=loss_pts,
        avg_r_win=avg_r_win, avg_r_loss=avg_r_loss,
        avg_rr_entry=t['rr'].mean(), expectancy_usd=exp,
        sharpe=sharpe, max_dd=max_dd, cagr=cagr, calmar=calmar,
        final_eq=curve[-1], net_pnl=curve[-1]-init_eq,
        avg_hold_min=hold_min, by_zone=by_zone,
        curve=curve, daily_ret=dr.values,
    )

# ═══════════════════════════════════════════════════════════════
# KERN-SIMULATION  (v3)
# ═══════════════════════════════════════════════════════════════
def simulate(df_5m, df_15m, df_1h,
             zones_5m, zones_15m_fvg,
             piv_5m, piv_15m, piv_1h) -> pd.DataFrame:

    def _bias5m(piv):
        b = compute_bias(piv)
        b = b[~b.index.duplicated(keep='last')]
        return _to_arr(b.reindex(df_5m.index, method='ffill').fillna('neutral'))
    b1h_arr  = _bias5m(piv_1h)
    b15m_arr = _bias5m(piv_15m)

    p1h = piv_1h[~piv_1h.index.duplicated(keep='last')]
    sh1h_arr = p1h['sh_val'].reindex(df_5m.index).ffill().values
    sl1h_arr = p1h['sl_val'].reindex(df_5m.index).ffill().values

    all_zones = sorted(zones_5m, key=lambda z: z['origin_bar'])
    zone_ptr  = 0; active: list = []

    htf_sorted = sorted(zones_15m_fvg, key=lambda z: z['origin_ts'])
    htf_ptr = 0; htf_bull: list = []; htf_bear: list = []

    sh_arr = piv_5m['sh_val'].values
    sl_arr = piv_5m['sl_val'].values
    run_sh: list = []; run_sl: list = []

    O = df_5m['open'].values; H = df_5m['high'].values
    L = df_5m['low'].values;  C = df_5m['close'].values
    N = len(df_5m); idx = df_5m.index

    trades: list = []
    in_trade = False; trade: dict = {}

    # Equity-Tracking  (peak_eq = echtes All-Time-High, wird NIEMALS zurückgesetzt)
    current_eq = INIT_EQ
    peak_eq    = INIT_EQ

    # Tages-Kontrollen
    daily_cnt:     dict = {}
    day_consec_sl: dict = {}
    consec_sl       = 0
    last_result_day = None

    # Sweep-State
    last_sweep_bar = -999
    last_sweep_dir = ''

    for i in range(N):
        ts = idx[i]

        # Pivots aktualisieren
        if not np.isnan(sh_arr[i]):
            run_sh.append(sh_arr[i])
            if len(run_sh) > EQL_LOOKBACK: run_sh.pop(0)
        if not np.isnan(sl_arr[i]):
            run_sl.append(sl_arr[i])
            if len(run_sl) > EQL_LOOKBACK: run_sl.pop(0)

        last_1h_sh = sh1h_arr[i]
        last_1h_sl = sl1h_arr[i]

        # Sweep-Detection
        if run_sl:
            ns = min(run_sl[-SWEEP_LOOKBACK:])
            if L[i] < ns and C[i] > ns:
                last_sweep_bar = i; last_sweep_dir = 'bull'
        if run_sh:
            ns = max(run_sh[-SWEEP_LOOKBACK:])
            if H[i] > ns and C[i] < ns:
                last_sweep_bar = i; last_sweep_dir = 'bear'

        # Zonen freigeben
        while zone_ptr < len(all_zones) and all_zones[zone_ptr]['origin_bar'] < i:
            active.append(copy.copy(all_zones[zone_ptr])); zone_ptr += 1

        while htf_ptr < len(htf_sorted) and htf_sorted[htf_ptr]['origin_ts'] < ts:
            z = htf_sorted[htf_ptr]
            (htf_bull if z['dir']=='bull' else htf_bear).append(
                z['top'] if z['dir']=='bull' else z['bottom'])
            htf_ptr += 1

        active = [z for z in active
                  if z['active'] and not z['used']
                  and (i - z['origin_bar']) <= ZONE_MAX_AGE]

        # ── Offenen Trade verwalten ────────────────────────────
        if in_trade:
            d = trade['dir']
            hit_sl = (d==1 and L[i]<=trade['sl']) or (d==-1 and H[i]>=trade['sl'])
            hit_tp = (d==1 and H[i]>=trade['tp']) or (d==-1 and L[i]<=trade['tp'])

            if hit_sl or hit_tp:
                ep  = trade['sl'] if hit_sl else trade['tp']
                pts = (ep - trade['entry']) * d
                pnl = pts * trade['contracts'] * NQ_PV
                trade.update(exit_ts=ts, exit_price=ep,
                             result='SL' if hit_sl else 'TP',
                             pnl_pts=pts, pnl_usd=pnl)
                trades.append(trade)

                # Equity updaten
                current_eq += pnl
                peak_eq     = max(peak_eq, current_eq)   # echtes ATH, kein Reset

                # Tages-SL-Streak
                trade_day = ts.date()
                if trade['result'] == 'SL':
                    if trade_day == last_result_day:
                        consec_sl += 1
                    else:
                        consec_sl = 1
                    day_consec_sl[trade_day] = max(
                        day_consec_sl.get(trade_day, 0), consec_sl)
                else:
                    consec_sl = 0
                last_result_day = trade_day
                in_trade = False
            continue

        # ── ENTRY-FILTER ───────────────────────────────────────

        hour = ts.hour + ts.minute / 60
        if not ((KZ1_START <= hour < KZ1_END) or (KZ2_START <= hour < KZ2_END)):
            continue

        day = ts.date()
        if daily_cnt.get(day, 0) >= MAX_TRADES_DAY:
            continue

        if day_consec_sl.get(day, 0) >= MAX_CONSEC_SL:
            continue

        b1h = b1h_arr[i]; b15m = b15m_arr[i]
        if b1h == 0 or b15m == 0 or b1h != b15m:
            continue
        pref = 'bull' if b1h == 1 else 'bear'

        if (i - last_sweep_bar) > SWEEP_VALID_BARS or last_sweep_dir != pref:
            continue

        fib_ok = False
        if not (np.isnan(last_1h_sh) or np.isnan(last_1h_sl)):
            rng_1h = last_1h_sh - last_1h_sl
            # OTE zone: 61.8 % discount (bull below 61.8 %, bear above 38.2 %)
            if pref == 'bull' and C[i] < last_1h_sl + 0.618 * rng_1h: fib_ok = True
            if pref == 'bear' and C[i] > last_1h_sh - 0.618 * rng_1h: fib_ok = True
        if not fib_ok:
            continue

        # IFVG-Flip
        for z in active:
            if z['type'] != 'FVG': continue
            if z['dir']=='bull' and C[i] < z['bottom']:
                z['dir']='bear'; z['type']='IFVG'
                z['entry']=z['bottom']; z['sl']=z['top']+SL_BUF
            elif z['dir']=='bear' and C[i] > z['top']:
                z['dir']='bull'; z['type']='IFVG'
                z['entry']=z['top']; z['sl']=z['bottom']-SL_BUF

        ksh = np.sort(run_sh) if run_sh else np.array([])
        ksl = np.sort(run_sl) if run_sl else np.array([])
        htf_b = np.sort(htf_bull) if htf_bull else np.array([])
        htf_r = np.sort(htf_bear) if htf_bear else np.array([])
        eqh, eql = recent_eql(run_sh, run_sl)

        # Dynamisches Risiko (Compound + Bremsen)
        risk_pct   = effective_risk(current_eq, peak_eq)
        # Safety-Cap: schlechtestmöglicher SL darf das 7%-Limit nicht überschreiten
        current_dd = (peak_eq - current_eq) / peak_eq
        headroom   = DD_MAX - current_dd                 # verbleibender DD-Spielraum bis 7%
        max_risk   = peak_eq * headroom / current_eq    # als Anteil von current_eq
        risk_pct   = min(risk_pct, max(max_risk, 0.0))
        if risk_pct < 0.0002: continue                  # < 0.02 % → nicht lohnenswert
        risk_usd   = current_eq * risk_pct

        for z in active:
            if z['dir'] != pref: continue

            entry_p = z['entry']; sl_p = z['sl']
            sl_dist = abs(entry_p - sl_p)
            if sl_dist > MAX_SL or sl_dist < 1.0:
                z['active'] = False; continue

            if pref == 'bull':
                if C[i] < z['bottom']-1.0: z['active']=False; continue
                touched = L[i] <= z['top'] and H[i] >= z['bottom']
            else:
                if C[i] > z['top']+1.0: z['active']=False; continue
                touched = H[i] >= z['bottom'] and L[i] <= z['top']
            if not touched: continue

            tdir   = 1 if pref=='bull' else -1
            min_tp = entry_p + tdir*MIN_RR*sl_dist

            if pref == 'bull':
                tp_p = min_tp
                ix = np.searchsorted(ksh, min_tp)
                if ix < len(ksh): tp_p = min(tp_p, ksh[ix])
                if len(eqh):
                    s=np.sort(eqh); ix2=np.searchsorted(s,min_tp)
                    if ix2<len(s): tp_p=min(tp_p,s[ix2])
                if len(htf_b):
                    ix3=np.searchsorted(htf_b,min_tp)
                    if ix3<len(htf_b): tp_p=min(tp_p,htf_b[ix3])
            else:
                tp_p = min_tp
                ix = np.searchsorted(ksl,min_tp,side='right')-1
                if ix>=0: tp_p=max(tp_p,ksl[ix])
                if len(eql):
                    s=np.sort(eql); ix2=np.searchsorted(s,min_tp,side='right')-1
                    if ix2>=0: tp_p=max(tp_p,s[ix2])
                if len(htf_r):
                    ix3=np.searchsorted(htf_r,min_tp,side='right')-1
                    if ix3>=0: tp_p=max(tp_p,htf_r[ix3])

            if abs(tp_p-entry_p)/sl_dist < MIN_RR: continue

            if i+1 >= N: continue
            fill      = O[i+1]
            sl_dist_f = abs(fill-sl_p)
            rr_f      = abs(tp_p-fill)/max(sl_dist_f,0.01)
            if rr_f < MIN_RR or sl_dist_f > MAX_SL: continue

            contracts = risk_usd / (sl_dist_f * NQ_PV)

            trade = dict(
                entry_ts=idx[i+1], exit_ts=None,
                entry=fill, sl=sl_p, tp=tp_p,
                dir=tdir, zone_type=z['type'],
                sl_dist=sl_dist_f, rr=rr_f,
                contracts=contracts,
                risk_pct=risk_pct, risk_usd=risk_usd,
                eq_at_entry=current_eq,
                exit_price=None, result=None,
                pnl_pts=None, pnl_usd=None,
            )
            z['used']  = True
            in_trade   = True
            daily_cnt[day] = daily_cnt.get(day, 0) + 1
            break

    if in_trade and trade:
        pts = (C[-1]-trade['entry'])*trade['dir']
        pnl = pts*trade['contracts']*NQ_PV
        trade.update(exit_ts=idx[-1],exit_price=C[-1],
                     result='EOD',pnl_pts=pts,pnl_usd=pnl)
        trades.append(trade)

    return pd.DataFrame(trades) if trades else pd.DataFrame()

# ═══════════════════════════════════════════════════════════════
# PHASE 1 – MONTE CARLO
# ═══════════════════════════════════════════════════════════════
def monte_carlo(trades: pd.DataFrame) -> dict:
    pnl = trades['pnl_usd'].values; n = len(pnl)
    rng = np.random.default_rng(42)
    curves = np.zeros((MC_SIMS,n+1)); curves[:,0] = INIT_EQ
    for s in range(MC_SIMS):
        curves[s,1:] = INIT_EQ+np.cumsum(rng.choice(pnl,size=n,replace=True))
    ends = curves[:,-1]; floor = INIT_EQ*0.90
    sharpes = []
    for s in range(MC_SIMS):
        r=np.diff(curves[s])/curves[s,:-1]
        sharpes.append(r.mean()/max(r.std(),1e-9)*np.sqrt(252))
    return dict(
        curves=curves,
        breach_rate=(curves.min(axis=1)<floor).mean(),
        profitable=(ends>INIT_EQ).mean(),
        p5=np.percentile(ends,5),p50=np.percentile(ends,50),p95=np.percentile(ends,95),
        sh_p5=np.percentile(sharpes,5),sh_p50=np.median(sharpes),sh_p95=np.percentile(sharpes,95),
    )

# ═══════════════════════════════════════════════════════════════
# PHASE 2 – WALK-FORWARD
# ═══════════════════════════════════════════════════════════════
def walk_forward(trades: pd.DataFrame, in_mo=18, oos_mo=6, step_mo=3) -> list:
    if len(trades)==0: return []
    t=trades.copy(); t['entry_ts']=pd.to_datetime(t['entry_ts'])
    t=t.set_index('entry_ts').sort_index()
    start,end=t.index.min(),t.index.max()
    results,cur=[],start
    while True:
        in_end=cur+pd.DateOffset(months=in_mo)
        oos_end=in_end+pd.DateOffset(months=oos_mo)
        if oos_end>end+pd.DateOffset(months=1): break
        ins=t.loc[cur:in_end]; oos=t.loc[in_end:oos_end]
        if len(ins)<5 or len(oos)<3:
            cur+=pd.DateOffset(months=step_mo); continue
        mi=calc_metrics(ins.reset_index()); mo=calc_metrics(oos.reset_index())
        results.append(dict(
            period=cur.strftime('%Y-%m'),
            in_n=mi['n_trades'],oos_n=mo['n_trades'],
            in_wr=mi['win_rate'],oos_wr=mo['win_rate'],
            in_pf=mi['profit_factor'],oos_pf=mo['profit_factor'],
            in_sh=mi['sharpe'],oos_sh=mo['sharpe'],
        ))
        cur+=pd.DateOffset(months=step_mo)
    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 3 – BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════
def bootstrap_ci(trades: pd.DataFrame, n_boot=2000) -> dict:
    pnl=trades['pnl_usd'].values; n=len(pnl)
    rng=np.random.default_rng(123)
    sharpes,pfs,wrs,dds=[],[],[],[]
    for _ in range(n_boot):
        starts=rng.integers(0,max(n-BLOCK_SZ+1,1),size=n//BLOCK_SZ+1)
        sample=np.concatenate([pnl[s:s+BLOCK_SZ] for s in starts])[:n]
        curve=np.concatenate([[INIT_EQ],INIT_EQ+np.cumsum(sample)])
        r=np.diff(curve)/curve[:-1]
        sharpes.append(r.mean()/max(r.std(),1e-9)*np.sqrt(252))
        pfs.append(sample[sample>0].sum()/max(-sample[sample<0].sum(),1e-9))
        wrs.append((sample>0).mean())
        dds.append(((curve-np.maximum.accumulate(curve))/np.maximum.accumulate(curve)).min())
    ci=lambda a: np.percentile(a,[2.5,97.5])
    return dict(
        sharpe_ci=ci(sharpes),sharpe_med=np.median(sharpes),
        pf_ci=ci(pfs),pf_med=np.median(pfs),
        wr_ci=ci(wrs),dd_ci=ci(dds),
    )

# ═══════════════════════════════════════════════════════════════
# PHASE 4 – PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════
def permutation_test(trades: pd.DataFrame) -> dict:
    pnl=trades['pnl_usd'].values; r=pnl/INIT_EQ
    true_sh=r.mean()/max(r.std(),1e-9)*np.sqrt(252)
    rng=np.random.default_rng(99)
    psh=np.array([rng.permutation(pnl/INIT_EQ) for _ in range(PERM_N)])
    psh=np.array([p.mean()/max(p.std(),1e-9)*np.sqrt(252) for p in psh])
    return dict(true_sh=true_sh,perm_sh=psh,
                p_value=(psh>=true_sh).mean(),
                perm_p95=np.percentile(psh,95),perm_med=np.median(psh))

# ═══════════════════════════════════════════════════════════════
# KONSOLEN-REPORT
# ═══════════════════════════════════════════════════════════════
def print_report(m, mc, wf, boot, perm, trades):
    S="═"*65
    print(f"\n{S}\n  PHASE 0 – BASE BACKTEST  (v3: Compound + DD-Brake)\n{S}")
    print(f"  Trades              : {m['n_trades']}")
    print(f"  Win Rate            : {m['win_rate']:.2%}")
    print(f"  Loss Rate           : {m['loss_rate']:.2%}")
    print(f"  Profit Factor       : {m['profit_factor']:.3f}")
    print(f"  Avg Win  (USD / pts): ${m['avg_win_usd']:>10,.1f}  /  {m['avg_win_pts']:+.1f} pts")
    print(f"  Avg Loss (USD / pts): ${m['avg_loss_usd']:>10,.1f}  /  {m['avg_loss_pts']:+.1f} pts")
    print(f"  Avg R  wins/losses  : {m['avg_r_win']:+.2f}R  /  {m['avg_r_loss']:+.2f}R")
    print(f"  Avg RR at entry     : {m['avg_rr_entry']:.2f}")
    print(f"  Expectancy/trade    : ${m['expectancy_usd']:,.1f}")
    print(f"  Avg hold time       : {m['avg_hold_min']:.0f} min")
    print(f"  Sharpe (daily, ann) : {m['sharpe']:.3f}")
    print(f"  Max Drawdown        : {m['max_dd']:.2%}")
    print(f"  CAGR                : {m['cagr']:.2%}")
    print(f"  Calmar Ratio        : {m['calmar']:.2f}")
    print(f"  Final Equity        : ${m['final_eq']:,.0f}  (net {m['net_pnl']:+,.0f})")

    rp = trades['risk_pct']*100
    full  = (rp >= BASE_RISK_PCT*100*0.9).sum()
    half  = ((rp > BASE_RISK_PCT*100*0.4) & (rp < BASE_RISK_PCT*100*0.9)).sum()
    quart = (rp <= BASE_RISK_PCT*100*0.4).sum()
    print(f"\n  Risiko-Verteilung (DD-Brake):")
    print(f"    Voll   0.50% : {full} Trades")
    print(f"    Halb   0.25% : {half} Trades")
    print(f"    Viertel 0.125%: {quart} Trades")

    print(f"\n  By Zone Type:")
    for zt,v in m['by_zone'].items():
        print(f"    {zt:6s}  n={v['n']:4d}  WR={v['wr']}  P&L={v['pnl']}  AvgR={v['avg_r']}")

    print(f"\n{S}\n  PHASE 1 – MONTE CARLO  (n={MC_SIMS})\n{S}")
    print(f"  Floor-breach rate (−10%) : {mc['breach_rate']:.2%}")
    print(f"  P(profitable)            : {mc['profitable']:.2%}")
    print(f"  Final equity  5th pct    : ${mc['p5']:,.0f}")
    print(f"  Final equity  median     : ${mc['p50']:,.0f}")
    print(f"  Final equity  95th pct   : ${mc['p95']:,.0f}")
    print(f"  Sharpe 5/50/95           : {mc['sh_p5']:.3f} / {mc['sh_p50']:.3f} / {mc['sh_p95']:.3f}")

    print(f"\n{S}\n  PHASE 2 – WALK-FORWARD  (18M in / 6M OOS / 3M step)\n{S}")
    if wf:
        df_wf=pd.DataFrame(wf)
        print(df_wf[['period','in_n','oos_n','in_wr','oos_wr',
                     'in_pf','oos_pf','in_sh','oos_sh']
                    ].to_string(index=False,float_format=lambda x:f"{x:.3f}"))
        pos=(df_wf['oos_sh']>0).sum()
        eff=df_wf['oos_sh'].mean()/max(abs(df_wf['in_sh'].mean()),1e-9)
        print(f"\n  OOS positive-Sharpe : {pos}/{len(df_wf)}")
        print(f"  OOS/IS efficiency   : {eff:.2f}")

    print(f"\n{S}\n  PHASE 3 – BOOTSTRAP CI  (block={BLOCK_SZ}, n=2000)\n{S}")
    print(f"  Sharpe 95% CI  : [{boot['sharpe_ci'][0]:.3f}, {boot['sharpe_ci'][1]:.3f}]  median={boot['sharpe_med']:.3f}")
    print(f"  PF     95% CI  : [{boot['pf_ci'][0]:.3f}, {boot['pf_ci'][1]:.3f}]  median={boot['pf_med']:.3f}")
    print(f"  WR     95% CI  : [{boot['wr_ci'][0]:.2%}, {boot['wr_ci'][1]:.2%}]")
    print(f"  MaxDD  95% CI  : [{boot['dd_ci'][0]:.2%}, {boot['dd_ci'][1]:.2%}]")

    print(f"\n{S}\n  PHASE 4 – PERMUTATION TEST  (n={PERM_N})\n{S}")
    print(f"  True Sharpe              : {perm['true_sh']:.4f}")
    print(f"  Permuted median Sharpe   : {perm['perm_med']:.4f}")
    print(f"  Permuted 95th pct        : {perm['perm_p95']:.4f}")
    print(f"  p-value                  : {perm['p_value']:.4f}")
    print(f"  Verdict (α=0.05)         : {'✓ SIGNIFICANT' if perm['p_value']<0.05 else '✗ not significant'}")
    print(S)

# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════
def plot_dashboard(trades, m, mc, wf, boot, perm):
    fig=plt.figure(figsize=(22,30)); fig.patch.set_facecolor('#0d0d0d')
    gs=gridspec.GridSpec(5,3,figure=fig,hspace=0.48,wspace=0.35)
    CL='#00d4ff'; CW='#00c853'; CR='#ff1744'; CT='#e0e0e0'; CA='#1a1a2e'

    def sty(ax,title=''):
        ax.set_facecolor(CA); ax.tick_params(colors=CT,labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#333355')
        ax.xaxis.label.set_color(CT); ax.yaxis.label.set_color(CT)
        if title: ax.set_title(title,color=CT,fontsize=9,fontweight='bold')

    curve=m['curve']; nc=len(curve)
    ax0=fig.add_subplot(gs[0,:])
    ax0.plot(curve/1000,color=CL,lw=1.3)
    ax0.fill_between(range(nc),INIT_EQ/1000,curve/1000,where=curve>=INIT_EQ,alpha=0.12,color=CW)
    ax0.fill_between(range(nc),INIT_EQ/1000,curve/1000,where=curve<INIT_EQ,alpha=0.12,color=CR)
    ax0.axhline(INIT_EQ/1000,color='#555',lw=0.8,ls='--')
    ax0.axhline(INIT_EQ*(1-DD_BRAKE_SOFT)/1000,   color='#ffcc00',lw=0.8,ls=':',alpha=0.7,label=f'−{DD_BRAKE_SOFT*100:.1f}% Bremse soft')
    ax0.axhline(INIT_EQ*(1-DD_BRAKE_HARD)/1000,   color='#ff8800',lw=0.8,ls=':',alpha=0.7,label=f'−{DD_BRAKE_HARD*100:.1f}% Bremse hard')
    ax0.axhline(INIT_EQ*(1-DD_MAX)/1000,            color=CR,      lw=1.2,ls='--',alpha=0.9,label=f'−{DD_MAX*100:.1f}% Hard Limit')
    ax0.legend(loc='upper left',fontsize=7,framealpha=0.3)
    ax0.set_ylabel('Equity ($k)')
    sty(ax0,'PHASE 0 – Base Equity Curve  (v3: Compound 0.5% + DD-Bremsen + 7% Safety-Cap)')
    info=(f"n={m['n_trades']}  WR={m['win_rate']:.1%}  PF={m['profit_factor']:.2f}  "
          f"Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.1%}  "
          f"MaxDD={m['max_dd']:.1%}  Calmar={m['calmar']:.2f}  Final=${m['final_eq']:,.0f}")
    ax0.text(0.01,0.04,info,transform=ax0.transAxes,color=CT,fontsize=8.5,
             bbox=dict(boxstyle='round',facecolor='#111133',alpha=0.8))

    ax1=fig.add_subplot(gs[1,0])
    pu=trades['pnl_usd'].values
    ax1.hist(pu[pu>0]/1000,bins=25,color=CW,alpha=0.75,label='Wins')
    ax1.hist(pu[pu<0]/1000,bins=25,color=CR,alpha=0.75,label='Losses')
    ax1.set_xlabel('P&L ($k)'); ax1.legend(fontsize=7,framealpha=0.3)
    sty(ax1,'P&L Distribution')

    ax2=fig.add_subplot(gs[1,1])
    zt=trades.groupby('zone_type')['pnl_usd'].agg(['count','sum'])
    ax2.bar(zt.index,zt['sum']/1000,color=[CW if v>=0 else CR for v in zt['sum']],alpha=0.85)
    for bar,(_,row) in zip(ax2.patches,zt.iterrows()):
        ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height(),
                 f"n={int(row['count'])}",ha='center',va='bottom',color=CT,fontsize=7)
    ax2.set_ylabel('Total P&L ($k)'); sty(ax2,'P&L by Zone Type')

    ax2b=fig.add_subplot(gs[1,2])
    rp=trades['risk_pct']*100
    rf=(rp>=BASE_RISK_PCT*100*0.9).sum()
    rh=((rp>BASE_RISK_PCT*100*0.4)&(rp<BASE_RISK_PCT*100*0.9)).sum()
    rq=(rp<=BASE_RISK_PCT*100*0.4).sum()
    sizes=[rf,rh,rq]; labels=[f'0.50% ({rf})',f'0.25% ({rh})',f'0.125% ({rq})']
    nonzero=[(s,l,'#00c853' if j==0 else '#ffcc00' if j==1 else '#ff1744')
             for j,(s,l) in enumerate(zip(sizes,labels)) if s>0]
    if nonzero:
        ax2b.pie([x[0] for x in nonzero],labels=[x[1] for x in nonzero],
                 colors=[x[2] for x in nonzero],autopct='%1.0f%%',
                 textprops={'color':CT,'fontsize':8})
    sty(ax2b,'Risk-Verteilung (DD-Brake)')

    ax4=fig.add_subplot(gs[2,:])
    crvs=mc['curves']
    for s in range(MC_SIMS): ax4.plot(crvs[s]/1000,color=CL,alpha=0.04,lw=0.5)
    p5=np.percentile(crvs,5,axis=0); p50=np.percentile(crvs,50,axis=0); p95=np.percentile(crvs,95,axis=0)
    ax4.plot(p50/1000,color='#ffcc00',lw=1.5,label='Median')
    ax4.fill_between(range(len(p5)),p5/1000,p95/1000,alpha=0.18,color=CL,label='5–95%')
    ax4.axhline(INIT_EQ/1000,color='#555',lw=0.8,ls='--')
    ax4.axhline(INIT_EQ*0.9/1000,color=CR,lw=0.8,ls=':',label='−10% floor')
    ax4.text(0.01,0.04,
             f"Breach: {mc['breach_rate']:.2%}  P(profit): {mc['profitable']:.2%}  "
             f"5th: ${mc['p5']:,.0f}  Median: ${mc['p50']:,.0f}  95th: ${mc['p95']:,.0f}",
             transform=ax4.transAxes,color=CT,fontsize=8.5,
             bbox=dict(boxstyle='round',facecolor='#111133',alpha=0.8))
    ax4.set_ylabel('Equity ($k)'); sty(ax4,'PHASE 1 – Monte Carlo 200 Simulations')
    ax4.legend(loc='upper left',fontsize=8,framealpha=0.3)

    ax5=fig.add_subplot(gs[3,:2])
    if wf:
        df_wf=pd.DataFrame(wf); x=range(len(df_wf))
        ax5.plot(x,df_wf['in_sh'],'o-',color='#ffcc00',lw=1.2,ms=5,label='IS Sharpe')
        ax5.plot(x,df_wf['oos_sh'],'s-',color=CL,lw=1.2,ms=5,label='OOS Sharpe')
        ax5.axhline(0,color='#555',lw=0.8,ls='--')
        ax5.set_xticks(list(x)); ax5.set_xticklabels(df_wf['period'].tolist(),rotation=45,fontsize=6)
        ax5.legend(fontsize=8,framealpha=0.3)
    sty(ax5,'PHASE 2 – Walk-Forward Sharpe')

    ax5b=fig.add_subplot(gs[3,2])
    if wf:
        df_wf=pd.DataFrame(wf); x=range(len(df_wf))
        ax5b.bar([xi-.2 for xi in x],df_wf['in_pf'],width=0.35,color='#ffcc00',alpha=0.85,label='IS')
        ax5b.bar([xi+.2 for xi in x],df_wf['oos_pf'],width=0.35,color=CL,alpha=0.85,label='OOS')
        ax5b.axhline(1.0,color='#555',lw=0.8,ls='--')
        ax5b.set_xticks(list(x)); ax5b.set_xticklabels(df_wf['period'].tolist(),rotation=45,fontsize=6)
        ax5b.legend(fontsize=7,framealpha=0.3)
    sty(ax5b,'WF Profit Factor')

    ax6=fig.add_subplot(gs[4,0])
    labels=['Sharpe','PF','WR','MaxDD']
    cis=[boot['sharpe_ci'],boot['pf_ci'],boot['wr_ci'],boot['dd_ci']]
    meds=[boot['sharpe_med'],boot['pf_med'],np.mean(boot['wr_ci']),np.mean(boot['dd_ci'])]
    for j,(ci,med,lbl) in enumerate(zip(cis,meds,labels)):
        ax6.barh(j,ci[1]-ci[0],left=ci[0],height=0.45,color=CL,alpha=0.6)
        ax6.plot(med,j,'o',color='#ffcc00',ms=6)
        ax6.text(ci[1]+0.001,j,f'{ci[0]:.2f}–{ci[1]:.2f}',va='center',color=CT,fontsize=7)
    ax6.set_yticks(range(4)); ax6.set_yticklabels(labels)
    sty(ax6,'PHASE 3 – Bootstrap 95% CI')

    ax7=fig.add_subplot(gs[4,1:])
    ps=perm['perm_sh']
    ax7.hist(ps,bins=60,color='#445566',alpha=0.85,density=True,label='Permuted Sharpe')
    ax7.axvline(perm['true_sh'],color='#ffcc00',lw=2.5,label=f"True={perm['true_sh']:.3f}")
    ax7.axvline(perm['perm_p95'],color=CL,lw=1.5,ls='--',label=f"95th={perm['perm_p95']:.3f}")
    col=CW if perm['p_value']<0.05 else CR
    ax7.text(0.60,0.87,f"p-value = {perm['p_value']:.4f}",
             transform=ax7.transAxes,color=col,fontsize=11,fontweight='bold',
             bbox=dict(boxstyle='round',facecolor='#111133',alpha=0.85))
    ax7.legend(fontsize=8,framealpha=0.3)
    sty(ax7,f'PHASE 4 – Permutation Test (n={PERM_N})')

    fig.suptitle(
        f'NQ ICT v3  ·  FVG/IFVG + Sweep + Kill Zones + Fib OTE  ·  '
        f'Compound 0.5%  ·  Bremsen @3.5/5.5%  ·  Safety-Cap @7%  ·  Max {MAX_TRADES_DAY}/Tag',
        color=CT,fontsize=10,fontweight='bold',y=0.999)

    out=OUT_DIR/'ict_v3_dashboard.png'
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(); print(f"Dashboard → {out}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("Loading data …")
    df_5m =load_nq('nq-5m.csv'); df_15m=load_nq('nq-15m.csv'); df_1h=load_nq('nq-1h.csv')
    print(f"  5M: {len(df_5m):,} | 15M: {len(df_15m):,} | 1H: {len(df_1h):,}")

    print("\nComputing pivots …")
    piv_5m=detect_pivots(df_5m,SWING_N_5M)
    piv_15m=detect_pivots(df_15m,SWING_N_15M)
    piv_1h=detect_pivots(df_1h,SWING_N_1H)

    print("Pre-computing FVG zones …")
    fvg_5m=precompute_fvg(df_5m); fvg_15m=precompute_fvg(df_15m)
    print(f"  5M FVG: {len(fvg_5m):,}  |  15M FVG: {len(fvg_15m):,}")

    print("\n[Phase 0] Simulation …")
    trades=simulate(df_5m,df_15m,df_1h,fvg_5m,fvg_15m,piv_5m,piv_15m,piv_1h)

    if len(trades)==0: print("  Keine Trades generiert."); return

    trades.to_csv(OUT_DIR/'ict_v3_trades.csv',index=False)
    print(f"  {len(trades)} Trades  →  results/ict_v3_trades.csv")

    m=calc_metrics(trades)
    print("[Phase 1] Monte Carlo …"); mc=monte_carlo(trades)
    print("[Phase 2] Walk-Forward …"); wf=walk_forward(trades)
    if wf: pd.DataFrame(wf).to_csv(OUT_DIR/'ict_v3_walkforward.csv',index=False)
    print("[Phase 3] Bootstrap CI …"); boot=bootstrap_ci(trades)
    print("[Phase 4] Permutation Test …"); perm=permutation_test(trades)

    print_report(m,mc,wf,boot,perm,trades)
    print("\nGenerating dashboard …"); plot_dashboard(trades,m,mc,wf,boot,perm)
    print("Done.")

if __name__=='__main__':
    main()
