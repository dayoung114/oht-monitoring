"""Offline analysis tools. No LLM, equipment control, or ground-truth access."""
import pandas as pd
from core import incidents, priority
from xai import shap_sensors, reconstruction

MEANINGS={'NTC_온도':'장비 내부 온도','CT1_전류':'PCB 입력단 전류','CT2_전류':'PCB 출력단 전류','CT3_전류':'구동모터 1 출력 전류','CT4_전류':'구동모터 2 출력 전류','CT_Std':'CT1~4 측정값 간 표본 표준편차'}

def evidence(row, shap):
    sv,_=shap_sensors(row,shap)
    if not sv.empty: return sv,'이상 판정에 참여한 ML 모델의 양수 SHAP'
    if row.get('LSTM_pred',0)==1:
        rec=reconstruction(row)
        if not rec.empty: return rec,'LSTM 센서별 복원 오차'
    return pd.DataFrame(),'설명 자료 부족'

def observed_elapsed(event, received):
    q=received[(received.equipment_id==event.equipment_id)&(received.session_id==event.session_id)&(received.datetime>=event.start)&(received.datetime<=event.end)&(received.prediction==1)]
    return max(0.,(q.datetime.max()-event.start).total_seconds()) if not q.empty else 0.

def analyze(event, data, shap, cutoff, cadence):
    """All queries are explicitly cut off; labels and case_type are unused."""
    cutoff=pd.Timestamp(cutoff)
    past=data.loc[data.datetime<=cutoff]
    same=past[(past.equipment_id==event.equipment_id)&(past.session_id==event.session_id)].sort_values('datetime')
    q=same[(same.datetime>=event.start)&(same.datetime<=event.end)&(same.prediction==1)]
    if q.empty: raise ValueError('해당 시각까지 수신한 사건 관측이 없습니다.')
    row=q.iloc[-1]; ev,source=evidence(row,shap)
    at_cutoff=incidents(same,cadence,cutoff)
    matching=at_cutoff.loc[at_cutoff.event_id==event.event_id]
    duration=float(matching.iloc[0].duration_s) if not matching.empty else float(event.duration_s)
    top=ev.iloc[0,0] if not ev.empty else None
    lines=[f'장비: {event.equipment_id} / 사건 시작: {event.start}',f'조회 기준: {cutoff} / 근거 관측: {row.datetime}',f'관측된 이상 구간 합계: {duration:.1f}초',f'설명 근거: {source}']
    trace=['현재 사건 조회','XAI 조회']
    if top:
        trace.append('센서 추이 조회: '+top)
        lines.append(f'우선 확인 대상: {MEANINGS[top]} ({top})')
        lines.append(f'현재 관측값: {row[top]:.4f} (원본 단위)')
        baseline=same.loc[same.datetime<event.start].tail(30)
        if not baseline.empty: lines.append(f'사건 전 최대 30개 관측 중앙값: {baseline[top].median():.4f} / 차이: {row[top]-baseline[top].median():+.4f}')
        else: lines.append('사건 이전 비교 관측이 없습니다.')
        if top!='CT_Std':
            for suffix in ['_Diff','_MA_5','_MA_30']:
                col=top+suffix if top!='NTC_온도' else 'NTC_Diff'
                if col in row:
                    lines.append(f'{col}: {row[col]:.4f}')
                if top=='NTC_온도': break
    historical=past[(past.equipment_id==event.equipment_id)&(past.datetime<event.start)]
    history=incidents(historical,cadence,min(cutoff,event.start))
    trace.append('같은 장비 과거 사건 조회')
    if history.empty: lines.append('비교할 이전 사건 기록이 없습니다. 과거 이상이 없었다는 의미는 아닙니다.')
    else:
        lines.append(f'보유 이전 관측 기간: {historical.datetime.min()} ~ {historical.datetime.max()} / 이전 사건 {len(history)}건')
        # Similarity means same leading explanation sensor, never same fault.
        matches=[]
        if top:
            for h in history.head(100).itertuples():
                hr=historical[(historical.session_id==h.session_id)&(historical.datetime>=h.start)&(historical.datetime<h.end)&(historical.prediction==1)]
                if hr.empty: continue
                he,_=evidence(hr.iloc[-1],shap)
                if not he.empty and he.iloc[0,0]==top: matches.append(f'{h.start} ({h.duration_s:.1f}초)')
        lines.append('최근 최대 100건 중 주요 확인 센서가 같은 이전 사건: '+(' / '.join(matches[:3]) if matches else '확인되지 않음 또는 설명 자료 부족'))
    daily=past.loc[past.datetime.dt.date==cutoff.date()]
    ranks=priority(incidents(daily,cadence,cutoff))
    rank=ranks.loc[ranks['장비']==event.equipment_id]
    if not rank.empty: lines.append(f'당일 전체 장비 점검 순위: {int(rank.iloc[0]["순위"])}위 / {rank.iloc[0]["정렬 근거"]}')
    trace.append('당일 전체 장비 점검 순위 조회')
    lines.append('확인 참고: 관련 센서 추이와 관측 이력을 확인하고 현장 점검 필요 여부를 관리자가 판단하세요. 매뉴얼·운전 조건이 없어 고장 원인이나 허용 범위 초과를 확정할 수 없습니다.')
    return {'text':'\n\n'.join(lines),'trace':trace,'top':top,'cutoff':cutoff,'row_time':row.datetime}

def update_reports(events, data, shap, cutoff, cadence, reports, wait=10., refresh=30.):
    past=data.loc[data.datetime<=cutoff]
    for e in events.itertuples():
        elapsed=observed_elapsed(e,past)
        if elapsed<wait: continue
        rows=past[(past.equipment_id==e.equipment_id)&(past.session_id==e.session_id)&(past.datetime>=e.start)&(past.datetime<=e.end)&(past.prediction==1)]
        if rows.empty: continue
        row=rows.sort_values('datetime').iloc[-1]
        old=reports.get(e.event_id)
        if old and row.datetime<=old['row_time']: continue
        ev,_=evidence(row,shap);top=ev.iloc[0,0] if not ev.empty else None
        if old and top==old['top'] and (row.datetime-old['row_time']).total_seconds()<refresh: continue
        result=analyze(e,data,shap,row.datetime,cadence)
        result['trigger']='10초 등 설정 대기시간 충족' if old is None else ('주요 확인 센서 변경' if top!=old['top'] else '갱신 간격 도달')
        result['updates']=1 if old is None else old.get('updates',1)+1
        reports[e.event_id]=result
    return reports
