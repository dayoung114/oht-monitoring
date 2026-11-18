import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import incidents, SENSORS
from analysis_support import update_reports, analyze

def sample():
    d=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=45,freq='s'),'equipment_id':'x','session_id':'1','prediction':1,'LSTM_pred':1})
    for i,s in enumerate(SENSORS):
        d[s]=i+1.;d['LSTM_error_'+s]=i+1.
    return d

def test_wait_refresh_pause_and_future():
    d=sample();reports={};shap=pd.DataFrame()
    for second,count in [(9,0),(10,1),(10,1),(39,1),(40,2)]:
        h=d.datetime.iloc[second];past=d[d.datetime<=h];events=incidents(past,1,h)
        update_reports(events,d,shap,h,1,reports,10,30)
        assert sum(r['updates'] for r in reports.values())==count
    r=next(iter(reports.values()))
    assert r['row_time']==d.datetime.iloc[40]
    # Future records and labels cannot affect analysis.
    h=d.datetime.iloc[10];e=incidents(d[d.datetime<=h],1,h).iloc[0]
    a=analyze(e,d,shap,h,1)
    changed=d.copy();changed.loc[changed.datetime>h,'CT_Std']=999999
    changed['y_true']=0;changed['case_type']='FN'
    assert analyze(e,changed,shap,h,1)==a

def test_short_event_manual():
    d=sample().iloc[:5];h=d.datetime.iloc[-1];e=incidents(d,1,h)
    assert update_reports(e,d,pd.DataFrame(),h,1,{},10,30)=={}
    assert analyze(e.iloc[0],d,pd.DataFrame(),h,1)['row_time']==h

def test_state_persists(tmp_path):
    from core import event_state
    db=tmp_path/'notes.sqlite3'
    assert event_state(db,'dataset','e')=='미확인'
    event_state(db,'dataset','e','확인 중')
    assert event_state(db,'dataset','e')=='확인 중'
    assert event_state(db,'other','e')=='미확인'
