"""Sensor-level display helpers. SHAP values are never mixed with MSE."""
import numpy as np
import pandas as pd
from core import SENSORS

def sensor_of(feature):
    if feature.startswith('NTC'): return 'NTC_온도'
    for s in SENSORS[1:5]:
        if feature.startswith(s): return s
    if feature=='CT_Std': return feature
    raise ValueError('연결할 수 없는 센서 피처: '+str(feature))

def reconstruction(row):
    vals=np.array([row.get('LSTM_error_'+s,np.nan) for s in SENSORS],dtype=float)
    if not np.isfinite(vals).all() or (vals<0).any(): return pd.DataFrame()
    total=vals.sum()
    return pd.DataFrame({'센서':SENSORS,'복원 오차':vals,'오차 비중(%)':vals/total*100 if total>0 else np.zeros(6)}).sort_values('복원 오차',ascending=False)

def shap_sensors(row, shap):
    """Within each firing ML model: sum positive feature SHAP by base sensor,
    normalize to 100%; then average available model shares. Not ensemble SHAP.
    """
    if shap.empty or 'eval_row' not in row: return pd.DataFrame(),pd.DataFrame()
    q=shap.loc[shap.eval_row==row.eval_row].copy()
    if q.empty: return pd.DataFrame(),q
    chunks=[]
    for model,g in q.groupby('model'):
        if row.get(model+'_pred',0)!=1: continue
        v=g.shap_value.clip(lower=0).groupby(g.feature.map(sensor_of)).sum().reindex(SENSORS,fill_value=0.)
        if v.sum()>0: chunks.append(v/v.sum()*100)
    if not chunks: return pd.DataFrame(),q
    vals=pd.concat(chunks,axis=1).mean(axis=1)
    return pd.DataFrame({'센서':vals.index,'이상 방향 기여 비중(%)':vals.values}).sort_values('이상 방향 기여 비중(%)',ascending=False),q
