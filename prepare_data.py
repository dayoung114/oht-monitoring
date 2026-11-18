"""Validate and join current analysis exports; never join on row order."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from core import KEY, SENSORS, MODELS, load_csv

def prepare(all_test: Path, shap_file: Path, sensor_file: Path, output: Path):
    x=pd.read_csv(all_test); raw=pd.read_csv(sensor_file); sh=pd.read_csv(shap_file)
    source_key=['장비ID','session_id','Datetime']
    for df in (x,raw):
        for c in source_key:
            if c not in df: raise ValueError(f'필수 키 누락: {c}')
        df['Datetime']=pd.to_datetime(df.Datetime,errors='raise')
        df['장비ID']=df['장비ID'].astype(str);df['session_id']=df.session_id.astype(str)
        if df.duplicated(source_key).any(): raise ValueError('관측 키 중복')
    if x.eval_row.isna().any() or not x.eval_row.is_unique: raise ValueError('eval_row 중복/결측')
    if not x[[m+'_pred' for m in MODELS]].isin([0,1]).all().all(): raise ValueError('이진 예측 오류')
    pred=x[[m+'_pred' for m in MODELS]].max(axis=1).astype(int)
    if not np.array_equal(pred,x.ensemble_pred): raise ValueError('1-of-4 판정 불일치')
    if not np.array_equal(x[[m+'_pred' for m in MODELS]].sum(axis=1),x.vote_count): raise ValueError('모델 예측 합 불일치')
    if '상태(state)' in raw:
        sensor_cols=source_key+[c for c in raw.columns if c not in source_key+['ID'] and c not in x.columns]+['상태(state)']
        sensor_cols=list(dict.fromkeys(sensor_cols))
    else: sensor_cols=source_key+SENSORS
    merged=x.merge(raw[sensor_cols],on=source_key,how='left',validate='one_to_one',indicator=True)
    if (merged._merge!='both').any(): raise ValueError('일부 예측 행의 원본 센서가 없습니다.')
    if '상태(state)' in merged and not np.array_equal(merged.y_true,merged['상태(state)']): raise ValueError('정답 라벨 불일치')
    merged=merged.drop(columns=['_merge']).rename(columns={'장비ID':'equipment_id','Datetime':'datetime','ensemble_pred':'prediction'})
    if sh.duplicated(['eval_row','model','feature']).any(): raise ValueError('SHAP 키 중복')
    if not set(sh.model).issubset({'XGB','RF','LR'}): raise ValueError('지원하지 않는 SHAP 모델')
    if not set(sh.eval_row).issubset(set(x.eval_row)): raise ValueError('연결되지 않는 SHAP 행')
    if not np.isfinite(sh[['feature_value','shap_value']].to_numpy()).all(): raise ValueError('SHAP 비유한값')
    # Validate SHAP feature values against original feature-engineered input.
    check=x[['eval_row']+source_key].merge(raw,on=source_key,validate='one_to_one').set_index('eval_row')
    for feature,g in sh.groupby('feature'):
        if feature not in check: raise ValueError(f'SHAP 피처가 원본에 없습니다: {feature}')
        if not np.allclose(check.loc[g.eval_row,feature].to_numpy(),g.feature_value.to_numpy(),rtol=1e-5,atol=1e-6):
            raise ValueError(f'SHAP 원본값 불일치: {feature}')
    output.mkdir(parents=True,exist_ok=True)
    csv=merged.to_csv(index=False).encode('utf-8-sig'); load_csv(csv)
    (output/'replay.csv').write_bytes(csv)
    sh.to_csv(output/'shap.csv',index=False,encoding='utf-8-sig')
    yt=x.y_true; yp=pred
    counts={k:int(v) for k,v in {'TN':((yt==0)&(yp==0)).sum(),'FP':((yt==0)&(yp==1)).sum(),'FN':((yt==1)&(yp==0)).sum(),'TP':((yt==1)&(yp==1)).sum()}.items()}
    report=dict(rows=len(merged),shap_points=int(sh.eval_row.nunique()),counts=counts,nominal_seconds=1,source='현재 업로드한 OHT_XAI_All_Test.csv',rule='1-of-4 OR',sensor_join='장비·세션·시간 1:1 검증',shap_feature_values='원본 파생변수 값 일치 확인')
    (output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all-test',required=True,type=Path);p.add_argument('--shap',required=True,type=Path);p.add_argument('--sensors',required=True,type=Path);p.add_argument('--output',default=Path('data'),type=Path);a=p.parse_args()
    prepare(a.all_test,a.shap,a.sensors,a.output)
