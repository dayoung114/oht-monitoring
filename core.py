"""Replay contract and incident accounting. No training or label-based prediction."""
from __future__ import annotations
import hashlib
import io
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

SENSORS = ['NTC_온도', 'CT1_전류', 'CT2_전류', 'CT3_전류', 'CT4_전류', 'CT_Std']
MODELS = ['LSTM', 'XGB', 'RF', 'LR']
KEY = ['equipment_id', 'session_id', 'datetime']

def load_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding='cp949')
    required = KEY + SENSORS + [m+'_pred' for m in MODELS]
    missing = sorted(set(required)-set(df.columns))
    if missing:
        raise ValueError('필수 컬럼 누락: '+', '.join(missing)+' / 안내서의 최신 결과 내보내기를 실행하세요.')
    if df.empty or df[KEY].isna().any().any():
        raise ValueError('빈 데이터 또는 장비·세션·시간 결측값이 있습니다.')
    for c in KEY[:2]:
        df[c] = df[c].astype(str)
    df['datetime'] = pd.to_datetime(df.datetime, errors='raise')
    if df.datetime.dt.tz is not None:
        df['datetime'] = df.datetime.dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    if df.duplicated(KEY).any():
        raise ValueError('장비·세션·시간 키가 중복됩니다.')
    for c in SENSORS:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if not np.isfinite(df[SENSORS].to_numpy()).all():
        raise ValueError('센서값에 NaN/Inf가 있습니다.')
    for m in MODELS:
        c = m+'_pred'
        if not df[c].isin([0,1]).all():
            raise ValueError(c+'는 결측 없는 0/1이어야 합니다.')
    pred = df[[m+'_pred' for m in MODELS]].max(axis=1).astype(int)
    if 'prediction' in df and not np.array_equal(df.prediction.to_numpy(), pred.to_numpy()):
        raise ValueError('저장된 최종 판정과 1-of-4 OR 결과가 다릅니다. 최신 결과로 다시 내보내세요.')
    df['prediction'] = pred
    evidence = ['evidence_'+s for s in SENSORS]
    if set(evidence).intersection(df.columns) and not set(evidence).issubset(df.columns):
        raise ValueError('센서 설명 컬럼 6개를 모두 제공해야 합니다.')
    if set(evidence).issubset(df.columns):
        df[evidence] = df[evidence].apply(pd.to_numeric, errors='raise')
        vals = df[evidence].to_numpy()
        if np.isinf(vals).any() or (vals[np.isfinite(vals)]<0).any():
            raise ValueError('센서 설명값은 0 이상의 유한값 또는 결측이어야 합니다.')
        partial = df[evidence].notna().any(axis=1) & ~df[evidence].notna().all(axis=1)
        if partial.any():
            raise ValueError('행별 센서 설명값이 일부만 존재합니다.')
    else:
        for c in evidence:
            df[c] = np.nan
    return df.sort_values(['datetime','equipment_id','session_id'], kind='stable').reset_index(drop=True)

def dataset_id(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def incidents(df: pd.DataFrame, nominal_seconds: float, horizon=None) -> pd.DataFrame:
    """Zero normal-gap tolerance. Break on session, calendar day, or missing interval.
    Each row represents [t,min(next t,t+cadence)). Last row capped at horizon.
    Only prior cadence is used; missing intervals never count as abnormal time.
    """
    cols = ['event_id','equipment_id','session_id','start','end','duration_s','points','status']
    if df.empty:
        return pd.DataFrame(columns=cols)
    if nominal_seconds <= 0:
        raise ValueError('측정 간격은 양수여야 합니다.')
    horizon = pd.Timestamp(horizon) if horizon is not None else df.datetime.max()
    records=[]
    for (eq,sess,day), g in df.groupby(['equipment_id','session_id',df.datetime.dt.date], sort=False):
        g = g.sort_values('datetime').reset_index(drop=True)
        current=None
        for i,row in g.iterrows():
            t=row.datetime
            nxt=g.iloc[i+1].datetime if i+1<len(g) else None
            contiguous=nxt is not None and (nxt-t).total_seconds() <= nominal_seconds*1.5
            end=min(t+pd.Timedelta(seconds=nominal_seconds),horizon,t.normalize()+pd.Timedelta(days=1))
            if nxt is not None:
                end=min(end,nxt)
            duration=max(0.,(end-t).total_seconds())
            if row.prediction == 1:
                if current is None:
                    event_id=f'{eq}|{sess}|{t.isoformat()}'
                    current=dict(event_id=event_id,equipment_id=eq,session_id=sess,start=t,end=end,duration_s=0.,points=0,status='종료')
                current['end']=end
                current['duration_s']+=duration
                current['points']+=1
                if nxt is None:
                    current['status']='진행 중' if horizon < t+pd.Timedelta(seconds=nominal_seconds) else '관측 종료'
                elif not contiguous:
                    current['status']='데이터 공백'
                if not contiguous or g.iloc[i+1].prediction==0:
                    records.append(current);current=None
            elif current is not None:
                records.append(current);current=None
        if current is not None:
            records.append(current)
    return pd.DataFrame(records,columns=cols).sort_values('start',ascending=False).reset_index(drop=True)

def priority(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=['순위','장비','총 이상 지속시간(초)','이상 사건 수','최근 이상 시각','정렬 근거'])
    out=events.groupby('equipment_id').agg(duration_s=('duration_s','sum'),count=('event_id','count'),latest=('start','max')).reset_index()
    out=out.sort_values(['duration_s','count','equipment_id'],ascending=[False,False,True]).reset_index(drop=True)
    ranks=[];last=None;rank=0
    for i,r in out.iterrows():
        pair=(r.duration_s,r['count'])
        if pair!=last: rank=i+1
        ranks.append(rank);last=pair
    out.insert(0,'순위',ranks)
    out['정렬 근거']=[f'총 이상 {s:.1f}초 · {n}건' for s,n in zip(out.duration_s,out['count'])]
    return out.rename(columns={'equipment_id':'장비','duration_s':'총 이상 지속시간(초)','count':'이상 사건 수','latest':'최근 이상 시각'})

def save_note(db: Path, dataset: str, event_id: str, text: str):
    db.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE IF NOT EXISTS notes (dataset TEXT,event TEXT,note TEXT,updated TEXT,PRIMARY KEY(dataset,event))')
        con.execute('INSERT OR REPLACE INTO notes VALUES (?,?,?,?)',(dataset,event_id,text,pd.Timestamp.now(tz='Asia/Seoul').isoformat()))

def read_note(db: Path,dataset: str,event_id: str) -> str:
    if not db.exists(): return ''
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE IF NOT EXISTS notes (dataset TEXT,event TEXT,note TEXT,updated TEXT,PRIMARY KEY(dataset,event))')
        r=con.execute('SELECT note FROM notes WHERE dataset=? AND event=?',(dataset,event_id)).fetchone()
    return r[0] if r else ''


def event_state(db, dataset, event_id, value=None):
    db.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE IF NOT EXISTS event_states (dataset TEXT,event TEXT,state TEXT,PRIMARY KEY(dataset,event))')
        if value is not None:
            if value not in ['미확인','확인 중','현장 점검 필요','조치 완료']: raise ValueError('처리 상태 오류')
            con.execute('INSERT OR REPLACE INTO event_states VALUES (?,?,?)',(dataset,event_id,value))
        row=con.execute('SELECT state FROM event_states WHERE dataset=? AND event=?',(dataset,event_id)).fetchone()
    return row[0] if row else '미확인'
