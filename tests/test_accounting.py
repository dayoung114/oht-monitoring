import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import pytest
from core import incidents, priority, load_csv, save_note, read_note, SENSORS

def rows(pred,secs=None,session=None):
    n=len(pred)
    return pd.DataFrame({'equipment_id':['A']*n,'session_id':session or ['s']*n,'datetime':[pd.Timestamp('2026-10-16')+pd.Timedelta(seconds=t) for t in (secs or list(range(n)))],'prediction':pred})

def test_no_normal_gap_tolerance():
    x=rows([1,1,0,1]);e=incidents(x,1,pd.Timestamp('2026-10-16 00:00:04'))
    assert len(e)==2 and e.duration_s.sum()==3

def test_no_future_time_counted():
    x=rows([1,1,1]);e=incidents(x,1,pd.Timestamp('2026-10-16 00:00:02.2'))
    assert len(e)==1 and e.duration_s.sum()==pytest.approx(2.2)
    assert e.iloc[0].status=='진행 중'

def test_session_and_missing_time_break():
    assert len(incidents(rows([1,1],session=['s1','s2']),1,pd.Timestamp('2026-10-16 00:00:02')))==2
    e=incidents(rows([1,1],[0,100]),1,pd.Timestamp('2026-10-16 00:01:41'))
    assert len(e)==2 and e.duration_s.sum()==2

def test_day_boundary_break_and_clip():
    d=rows([1,1]);d['datetime']=pd.to_datetime(['2026-10-16 23:59:59','2026-10-17 00:00:00'])
    e=incidents(d,1,pd.Timestamp('2026-10-17 00:00:01'))
    assert len(e)==2 and e.duration_s.sum()==2

def test_priority_and_ties():
    e=pd.DataFrame({'equipment_id':['A','B','B','C'],'duration_s':[5.,2.,3.,5.],'event_id':['a','b','c','d'],'start':[pd.Timestamp('2026-01-01')]*4})
    r=priority(e)
    assert r['장비'].tolist()==['B','A','C'] and r['순위'].tolist()==[1,2,2]

def test_reject_old_voting_and_duplicates():
    d=rows([0]);d['LSTM_pred']=1
    for m in ['XGB','RF','LR']:d[m+'_pred']=0
    for c in SENSORS:d[c]=0.
    with pytest.raises(ValueError,match='1-of-4'):load_csv(d.to_csv(index=False).encode())
    d.prediction=1;d=pd.concat([d,d])
    with pytest.raises(ValueError,match='중복'):load_csv(d.to_csv(index=False).encode())

def test_notes_separate_datasets(tmp_path):
    p=tmp_path/'notes.sqlite3';save_note(p,'d1','e1','점검 완료')
    assert read_note(p,'d1','e1')=='점검 완료'
    assert read_note(p,'d2','e1')==''

def test_real_data_final_metrics():
    p=Path(__file__).resolve().parents[1]/'data/replay.csv';d=load_csv(p.read_bytes())
    assert len(d)==9850
    y=d.y_true;z=d.prediction
    assert [sum((y==0)&(z==0)),sum((y==0)&(z==1)),sum((y==1)&(z==0)),sum((y==1)&(z==1))]==[5816,126,95,3813]

def test_all_abnormal_shap_coverage():
    root=Path(__file__).resolve().parents[1]
    d=load_csv((root/'data/replay.csv').read_bytes());s=pd.read_csv(root/'data/shap.csv')
    expected=set(d.loc[d.prediction==1,'eval_row'])
    assert len(expected)==3939 and set(s.eval_row)==expected
    assert len(s)==3939*3*19
    assert s.groupby(['eval_row','model']).size().eq(19).all()
    assert s.groupby('eval_row').model.nunique().eq(3).all()
    assert not s.duplicated(['eval_row','model','feature']).any()
