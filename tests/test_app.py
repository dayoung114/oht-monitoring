import sys
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def find(at,kind,label):
    return next(x for x in getattr(at,kind) if x.label==label)

def test_app_replay_and_history():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    assert not at.exception
    assert next(m.value for m in at.metric if m.label=='수신 장비')=='0'
    find(at,'button','다음 관측 시각').click().run()
    assert not at.exception
    # Live graph has no sensor samples beyond virtual receive time.
    assert at.session_state['clock'] is not None
    find(at,'radio','화면 모드').set_value('날짜별 조회').run()
    assert not at.exception
    assert len(at.get('plotly_chart'))>=4
    find(at,'selectbox','장비').set_value('oht01').run()
    assert not at.exception
    assert next(m.value for m in at.metric if m.label=='수신 장비')=='1'
    # Select the event containing a provided SHAP sample and verify the second XAI panel.
    from core import load_csv,incidents
    d=load_csv((ROOT/'data/replay.csv').read_bytes())
    target=d.loc[d.eval_row==149].iloc[0]
    e=incidents(d[(d.equipment_id==target.equipment_id)&(d.datetime.dt.date==target.datetime.date())],1,d.datetime.max()+pd.Timedelta(seconds=1))
    event=e[(e.start<=target.datetime)&(e.end>target.datetime)].iloc[0]
    find(at,'selectbox','분석할 이상 사건').set_value(event.event_id).run()
    find(at,'selectbox','사건 내 확인 시점').set_value(int(d.index[d.eval_row==149][0])).run()
    assert not at.exception
    assert len(at.get('plotly_chart'))>=6
    # LSTM-only anomaly: explain with reconstruction instead of a missing-SHAP warning.
    target=d.loc[(d.LSTM_pred==1)&(d[['XGB_pred','RF_pred','LR_pred']].sum(axis=1)==0)].iloc[0]
    find(at,'selectbox','장비').set_value(target.equipment_id).run()
    e=incidents(d[(d.equipment_id==target.equipment_id)&(d.datetime.dt.date==target.datetime.date())],1,d.datetime.max()+pd.Timedelta(seconds=1))
    event=e[(e.start<=target.datetime)&(e.end>target.datetime)].iloc[0]
    find(at,'selectbox','분석할 이상 사건').set_value(event.event_id).run()
    find(at,'selectbox','사건 내 확인 시점').set_value(int(target.name)).run()
    assert not at.exception
    assert any('이 시점의 이상 판정 근거는 센서별 복원 오차' in x.value for x in at.info)


def test_manual_analysis_ui():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    find(at,'radio','화면 모드').set_value('날짜별 조회').run()
    find(at,'button','선택 사건 지금 분석').click().run()
    assert not at.exception
    assert len(at.session_state['reports'])==1
    find(at,'button','근거 조회').click().run()
    assert not at.exception

def test_automatic_analysis_ui():
    from core import load_csv,incidents
    d=load_csv((ROOT/'data/replay.csv').read_bytes())
    first_day=d.datetime.dt.date.min()
    daily=d[d.datetime.dt.date==first_day]
    events=incidents(daily,1,daily.datetime.max())
    e=events[events.duration_s>12].sort_values('start').iloc[0]
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    at.session_state['clock']=e.start+pd.Timedelta(seconds=10)
    at.run()
    assert not at.exception
    assert e.event_id in at.session_state['reports']
    count=at.session_state['reports'][e.event_id]['updates']
    at.run()
    assert not at.exception
    assert at.session_state['reports'][e.event_id]['updates']==count
