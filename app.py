from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from core import SENSORS, KEY, MODELS, load_csv, dataset_id, incidents, priority, save_note, read_note, event_state
from xai import reconstruction, shap_sensors
from analysis_support import analyze, update_reports, MEANINGS

BASE=Path(__file__).resolve().parent
st.set_page_config(page_title='OHT 이상탐지 관제',layout='wide',page_icon='🏭')
st.markdown('''<style>
.stApp {background:#f5f7fb;} [data-testid="stMetric"] {background:white;border:1px solid #e1e7ef;padding:16px;border-radius:12px;}
h1 {letter-spacing:-1px;} [data-testid="stSidebar"] {background:#eaf0f7;}
</style>''',unsafe_allow_html=True)
st.title('OHT 이상탐지 관제')

@st.cache_data(show_spinner=False)
def cached_data(raw): return load_csv(raw)

with st.sidebar:
    st.subheader('조회 및 재생 설정')
try:
    raw=(BASE/'data/replay.csv').read_bytes()
    data=cached_data(raw)
except Exception as e:
    st.error(f'데이터를 열 수 없습니다: {e}');st.stop()
fingerprint=dataset_id(raw)
try:
    shap=pd.read_csv(BASE/'data/shap.csv')
    if not shap.empty:
        needed={'eval_row','model','feature','feature_value','shap_value'}
        if not needed.issubset(shap): raise ValueError('필수 SHAP 컬럼이 없습니다.')
        if 'eval_row' not in data or not data.eval_row.is_unique: raise ValueError('관제 CSV의 eval_row가 없거나 중복됩니다.')
        if not set(shap.eval_row).issubset(set(data.eval_row)): raise ValueError('관제 CSV와 연결되지 않는 eval_row가 있습니다.')
        if shap.duplicated(['eval_row','model','feature']).any(): raise ValueError('SHAP 키 중복')
        if not set(shap.model).issubset({'XGB','RF','LR'}): raise ValueError('SHAP 모델 이름 오류')
        if not np.isfinite(shap[['shap_value','feature_value']].to_numpy(dtype=float)).all(): raise ValueError('SHAP 결측/비유한값')
        # Match SHAP feature values to the bundled sensor input.
        check=data.set_index('eval_row')
        for feature,g in shap.groupby('feature'):
            if feature not in data: raise ValueError(f'SHAP 검증용 원본 피처가 없습니다: {feature}')
            if not np.allclose(check.loc[g.eval_row,feature].to_numpy(dtype=float),g.feature_value.to_numpy(dtype=float),rtol=1e-5,atol=1e-6):
                raise ValueError('SHAP 원본 피처값 불일치')
except Exception as e:
    st.warning(f'SHAP 파일을 사용하지 않습니다: {e}');shap=pd.DataFrame()

ss=st.session_state
if ss.get('dataset')!=fingerprint:
    ss.dataset=fingerprint;ss.running=False;ss.clock=None;ss.last_tick=time.monotonic();ss.alerted=set();ss.replay_token=None
with st.sidebar:
    mode=st.radio('화면 모드',['실시간 관제','날짜별 조회'])
    dates=sorted(data.datetime.dt.date.unique())
    day=st.selectbox('조회 날짜',dates,format_func=lambda d:d.strftime('%Y-%m-%d'))
    daily=data.loc[data.datetime.dt.date==day].copy()
    equipment=st.selectbox('장비',['전체']+sorted(daily.equipment_id.unique().tolist()))
    session_source=daily if equipment=='전체' else daily.loc[daily.equipment_id==equipment]
    session=st.selectbox('세션',['전체']+sorted(session_source.session_id.unique().tolist()))
    rate=st.select_slider('재생 속도',[1,2,5,10,30,60],value=5,format_func=lambda x:f'{x}배')
    skip_gaps=st.checkbox('관측이 없는 시간 건너뛰기',value=True,help='기록 사이의 긴 대기시간만 압축합니다. 센서값과 사건 지속시간은 원래 시각으로 계산합니다.')
    with st.expander('집계 기준'):
        cadence=st.number_input('측정 간격(초)',min_value=0.01,max_value=3600.,value=1.,step=0.1,help='기본 데이터는 같은 장비·세션 내 1초 간격임을 확인했습니다.')

with st.sidebar:
    with st.expander('자동 분석 설정', expanded=True):
        analysis_wait=st.number_input('자동 분석 대기시간(초)',min_value=0.,value=10.,step=1.)
        analysis_refresh=st.number_input('재분석 간격(초)',min_value=1.,value=30.,step=1.)

start=daily.datetime.min();finish=daily.datetime.max()+pd.Timedelta(seconds=cadence)
token=(fingerprint,str(day))
if ss.replay_token!=token:
    ss.replay_token=token;ss.running=False;ss.clock=start-pd.Timedelta(microseconds=1);ss.last_tick=time.monotonic();ss.alerted=set()
if mode=='날짜별 조회': ss.running=False

if mode=='실시간 관제':
    cols=st.columns(4)
    if cols[0].button('▶ 시작 / 계속',width="stretch"):
        if ss.clock>=finish: ss.clock=start-pd.Timedelta(microseconds=1);ss.alerted=set()
        ss.running=True;ss.last_tick=time.monotonic()
    if cols[1].button('Ⅱ 일시정지',width="stretch"): ss.running=False
    if cols[2].button('↺ 처음으로',width="stretch"):
        ss.running=False;ss.clock=start-pd.Timedelta(microseconds=1);ss.alerted=set()
    if cols[3].button('다음 관측 시각',width="stretch"):
        ss.running=False
        future=daily.loc[daily.datetime>ss.clock,'datetime']
        ss.clock=future.min() if not future.empty else finish


def chart_for(g,ev,center,before,after,chosen):
    fig=make_subplots(rows=len(chosen),cols=1,shared_xaxes=True,vertical_spacing=0.035,subplot_titles=chosen)
    for i,s in enumerate(chosen,1):
        # A gap in recorded data is drawn as a break, never a continuous trend.
        xx=[];yy=[];last=None
        for t,v in zip(g.datetime,g[s]):
            if last is not None and (t-last).total_seconds()>cadence*1.5: xx.append(None);yy.append(None)
            xx.append(t);yy.append(v);last=t
        fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',line=dict(color='#2463b4',width=2),name=s,showlegend=False,connectgaps=False),row=i,col=1)
        bad=g.loc[g.prediction==1]
        fig.add_trace(go.Scatter(x=bad.datetime,y=bad[s],mode='markers',marker=dict(color='#e34a4a',size=6),name='이상 판정',showlegend=i==1),row=i,col=1)
        for r in ev.itertuples():
            fig.add_vrect(x0=r.start,x1=r.end,fillcolor='#ec5a5a',opacity=.1,line_width=0,row=i,col=1)
        fig.add_vline(x=center,line_dash='dot',line_color='#e34a4a',row=i,col=1)
        fig.update_yaxes(title_text='센서값',row=i,col=1)
    lo=max(g.datetime.min(),center-pd.Timedelta(seconds=before)); hi=min(g.datetime.max()+pd.Timedelta(seconds=cadence),center+pd.Timedelta(seconds=after))
    fig.update_xaxes(range=[lo,hi],title_text='데이터 시각',row=len(chosen),col=1)
    fig.update_layout(height=max(340,210*len(chosen)),margin=dict(t=35,l=20,r=20,b=20),hovermode='x unified',dragmode='zoom',uirevision=str((str(center),before,after,chosen)))
    return fig


def render_report(report, question=None):
    parts=report['text'].split('\n\n')
    def value(prefix):
        return next((x[len(prefix):] for x in parts if x.startswith(prefix)), '—')
    def show_sensor():
        target=value('우선 확인 대상: ')
        with st.container(border=True):
            st.markdown('#### 우선 확인 대상')
            st.markdown(f'**{target}**')
            a,b=st.columns(2)
            a.metric('관측값',value('현재 관측값: ').replace(' (원본 단위)',''))
            base=value('사건 전 최대 30개 관측 중앙값: ')
            if base!='—':
                fields=base.split(' / 차이: ')
                b.metric('사건 전 중앙값',fields[0])
                st.write('중앙값 대비 변화: **'+fields[-1]+'**')
            else: b.metric('사건 전 중앙값','자료 없음')
            derived=[x.split(': ',1) for x in parts if any(x.startswith(k) for k in ['NTC_Diff:','CT1_전류_','CT2_전류_','CT3_전류_','CT4_전류_'])]
            if derived:
                names={'Diff':'직전 관측 대비 변화','MA_5':'최근 5개 관측 평균','MA_30':'최근 30개 관측 평균'}
                rows=[{'지표':next((v for k,v in names.items() if k in col),col),'값':val} for col,val in derived]
                st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.markdown('**판단 근거:** '+value('설명 근거: '))
    def show_history():
        with st.container(border=True):
            st.markdown('#### 과거 이력')
            found=[x for x in parts if x.startswith(('보유 이전 관측 기간:','최근 최대 100건 중'))]
            if found:
                for x in found: st.write(x)
            else: st.write('비교 가능한 이전 사건 기록 없음')
    def show_rank():
        with st.container(border=True):
            st.markdown('#### 점검 우선순위')
            st.write(value('당일 전체 장비 점검 순위: '))
    with st.container(border=True):
        cols=st.columns(3)
        cols[0].metric('장비',value('장비: ').split(' / ')[0])
        cols[1].metric('이상 지속시간',value('관측된 이상 구간 합계: '))
        cols[2].metric('분석 시각',str(report['cutoff'].strftime('%H:%M:%S')))
        st.write('발생 시각: '+value('장비: ').split(' / 사건 시작: ')[-1])
    if question and '이전에도' in question: show_history()
    elif question and '점검 순위' in question: show_rank()
    elif question: show_sensor()
    else:
        show_sensor()
        show_rank()
        show_history()
    with st.expander('분석 상세'):
        st.write('기준 시각: '+str(report['cutoff']))
        st.write('근거 관측: '+str(report['row_time']))
        st.write('단위: 원본 데이터 기준')
        st.write('현장 확인 필요 · 고장 원인 미확정')

@st.fragment(run_every=1.)
def dashboard():
    if mode=='실시간 관제' and ss.running:
        now=time.monotonic(); elapsed=max(0,now-ss.last_tick);ss.last_tick=now
        ss.clock=min(finish,ss.clock+pd.Timedelta(seconds=elapsed*rate))
        future=daily.loc[daily.datetime>ss.clock,'datetime']
        # Compress idle periods only when no new observation is about to arrive.
        if skip_gaps and not future.empty and (future.min()-ss.clock).total_seconds()>max(5.,rate):
            ss.clock=future.min()
        if ss.clock>=finish: ss.running=False
    horizon=finish if mode=='날짜별 조회' else ss.clock
    report_token=(token,str(analysis_wait),str(analysis_refresh),float(cadence))
    if ss.get('report_token')!=report_token or horizon<ss.get('report_horizon',horizon):
        ss.reports={};ss.report_token=report_token
    ss.report_horizon=horizon
    visible=daily.loc[daily.datetime<=horizon].copy()
    scope=visible
    if equipment!='전체': scope=scope.loc[scope.equipment_id==equipment]
    if session!='전체': scope=scope.loc[scope.session_id==session]
    events=incidents(scope,float(cadence),horizon)
    label='날짜 전체 기록' if mode=='날짜별 조회' else ('재생 중' if ss.running else '재생 대기 / 정지')
    st.markdown(f'{label} · 데이터 기준 시각: {horizon.strftime("%Y-%m-%d %H:%M:%S")} · 장비: {equipment} / 세션: {session}')
    if mode=='실시간 관제':
        st.progress(len(visible)/len(daily),text=f'날짜 전체 데이터 {len(visible):,} / {len(daily):,}행 수신')
    latest=scope.sort_values('datetime').groupby('equipment_id').tail(1)
    fresh=latest.loc[(horizon-latest.datetime).dt.total_seconds()<cadence*1.5]
    active=int(fresh.prediction.sum())
    cols=st.columns(4)
    cols[0].metric('수신 장비',scope.equipment_id.nunique())
    cols[1].metric('현재 이상 장비' if mode=='실시간 관제' else '이상 발생 장비',active if mode=='실시간 관제' else events.equipment_id.nunique())
    cols[2].metric('이상 사건',len(events))
    cols[3].metric('총 이상 지속시간',f'{events.duration_s.sum():,.1f}초' if not events.empty else '0초')
    if mode=='실시간 관제' and not events.empty:
        new=set(events.event_id)-ss.alerted
        if new:
            st.toast(f'새 이상 사건 {len(new)}건이 확인됐습니다.',icon='🚨');ss.alerted.update(new)
    all_events=incidents(visible,float(cadence),horizon)
    if mode=='실시간 관제':
        update_reports(all_events,data,shap,horizon,float(cadence),ss.reports,analysis_wait,analysis_refresh)
    tabs=st.tabs(['관제 · 점검 우선순위','이상 상세 · 센서 / XAI','요약 · 통계 · 분포','관리자 메모','자동 분석 · 질문'])
    with tabs[0]:
        st.subheader('점검 우선순위')
        rank=priority(events)
        if rank.empty: st.info('현재까지 집계된 이상 사건이 없습니다.')
        else: st.dataframe(rank,hide_index=True,width="stretch")
        st.subheader('장비 최근 상태')
        if latest.empty: st.info('시작 버튼을 누르면 센서 데이터가 들어옵니다.')
        else:
            status=latest[['equipment_id','session_id','datetime','prediction']].copy()
            status['상태']=np.where((horizon-status.datetime).dt.total_seconds()>=cadence*1.5,'새 데이터 대기',np.where(status.prediction==1,'이상','정상'))
            status=status.drop(columns='prediction').rename(columns={'equipment_id':'장비','session_id':'세션','datetime':'최근 수신 시각'})
            st.dataframe(status,hide_index=True,width="stretch")
        st.subheader('이상 이력')
        if not events.empty:
            view=events.rename(columns={'equipment_id':'장비','session_id':'세션','start':'시작','end':'관측 구간 끝','duration_s':'지속시간(초)','points':'이상 관측 수','status':'상태'}).drop(columns='event_id')
            st.dataframe(view,hide_index=True,width="stretch")
            st.download_button('이상 이력 CSV 다운로드',view.to_csv(index=False).encode('utf-8-sig'),file_name=f'OHT_events_{day}.csv',mime='text/csv')
    selected=None
    with tabs[1]:
        if scope.empty:
            st.info('아직 수신된 데이터가 없습니다.')
        else:
            if not events.empty:
                event_ids=events.event_id.tolist(); mapping=events.set_index('event_id',drop=False)
                event_id=st.selectbox('분석할 이상 사건',event_ids,format_func=lambda k:f'{mapping.loc[k,"equipment_id"]} · {mapping.loc[k,"start"]} · {mapping.loc[k,"duration_s"]:.1f}초',key='event_choice')
                selected=mapping.loc[event_id]
                eg=scope.loc[(scope.equipment_id==selected.equipment_id)&(scope.session_id==selected.session_id)].sort_values('datetime')
                candidates=eg.loc[(eg.datetime>=selected.start)&(eg.datetime<selected.end+pd.Timedelta(microseconds=1))&(eg.prediction==1)]
                if candidates.empty: candidates=eg.loc[eg.datetime==selected.start]
                row_ids=candidates.index.tolist()
                point=st.selectbox('사건 내 확인 시점',row_ids,format_func=lambda i:scope.loc[i,'datetime'].strftime('%H:%M:%S'),key='point_choice')
                row=scope.loc[point]; center=row.datetime
            else:
                eq=st.selectbox('센서를 볼 장비',sorted(scope.equipment_id.unique()),key='normal_eq')
                eg=scope.loc[scope.equipment_id==eq].sort_values('datetime')
                latest_session=eg.iloc[-1].session_id;eg=eg.loc[eg.session_id==latest_session]
                row=eg.iloc[-1];center=row.datetime
                st.info('현재 범위에 이상 사건이 없어 최근 센서 추이를 보여드립니다.')
            rec=reconstruction(row); sv,shap_point=shap_sensors(row,shap)
            preferred=sv['센서'].head(2).tolist() if not sv.empty else (rec['센서'].head(2).tolist() if not rec.empty else [])
            if row.prediction==1 and preferred:
                text=' · '.join(preferred)
                st.markdown(f'**우선 확인 센서: {text}**')
            controls=st.columns(3)
            before=controls[0].number_input('이상 전 표시 범위(초)',min_value=1,max_value=86400,value=60,step=10)
            after=controls[1].number_input('이상 후 표시 범위(초)',min_value=1,max_value=86400,value=60,step=10)
            chosen=controls[2].multiselect('표시 센서',SENSORS,default=SENSORS,key='sensors')
            ev=events.loc[(events.equipment_id==row.equipment_id)&(events.session_id==row.session_id)]
            if chosen:
                fig=chart_for(eg,ev,center,before,after,chosen)
                st.plotly_chart(fig,width="stretch",key='sensor_graph',config={'scrollZoom':True,'displaylogo':False})
            st.subheader('센서별 설명 (XAI)')
            left,right=st.columns(2)
            with left:
                st.markdown('**센서별 복원 오차 비중**')
                if rec.empty: st.info('해당 시점의 복원 오차가 제공되지 않았습니다.')
                else:
                    st.plotly_chart(px.bar(rec,x='오차 비중(%)',y='센서',orientation='h',range_x=[0,100],color_discrete_sequence=['#4b79b8']).update_layout(yaxis_autorange='reversed',height=300),width="stretch",key='rec_chart')
            with right:
                st.markdown('**센서별 이상 방향 기여 비중**')
                if sv.empty:
                    if not shap_point.empty and row.get('LSTM_pred',0)==1 and sum(row.get(m+'_pred',0) for m in ['XGB','RF','LR'])==0:
                        st.info('이 시점의 이상 판정 근거는 센서별 복원 오차입니다. 왼쪽 그래프를 확인하세요.')
                    elif not shap_point.empty:
                        st.info('SHAP 계산은 완료됐으나, 현재 표시 기준에 해당하는 양수 기여가 없습니다.')
                    elif row.prediction==0:
                        st.info('정상 판정 시점입니다. 이번 SHAP 계산 범위는 이상 판정 시점 전체입니다.')
                    else:
                        st.info('이 시점의 SHAP 데이터가 연결되지 않았습니다.')
                else:
                    st.plotly_chart(px.bar(sv,x='이상 방향 기여 비중(%)',y='센서',orientation='h',range_x=[0,100],color_discrete_sequence=['#d8834c']).update_layout(yaxis_autorange='reversed',height=300),width="stretch",key='shap_chart')
            if row.prediction==1 and selected is not None and preferred:
                earlier=eg.loc[eg.datetime<selected.start].tail(30)
                if not earlier.empty:
                    s=preferred[0];change=float(row[s]-earlier[s].median());direction='높습니다' if change>0 else ('낮습니다' if change<0 else '같습니다')
                    st.dataframe(pd.DataFrame([{'센서':s,'선택 시점 값':round(float(row[s]),3),'사건 전 중앙값':round(float(earlier[s].median()),3),'차이':round(change,3)}]),hide_index=True,width='stretch')
    with tabs[2]:
        if scope.empty: st.info('집계할 데이터가 없습니다.')
        else:
            st.subheader('장비별 이상 지속시간')
            if not events.empty:
                r=priority(events)
                st.plotly_chart(px.bar(r,x='장비',y='총 이상 지속시간(초)',color_discrete_sequence=['#406db1']),width="stretch",key='duration_chart')
            st.subheader('시간대별 이상 사건 시작 횟수')
            if not events.empty:
                h=events.groupby(events.start.dt.hour).size().reindex(range(24),fill_value=0).rename_axis('시간').reset_index(name='사건 수')
                st.plotly_chart(px.bar(h,x='시간',y='사건 수'),width="stretch",key='hour_chart')
            s=st.selectbox('분포를 볼 센서',SENSORS,key='dist_sensor')
            dist=scope[[s,'prediction']].copy();dist['판정']=dist.prediction.map({0:'정상',1:'이상'})
            st.plotly_chart(px.histogram(dist,x=s,color='판정',barmode='overlay',opacity=.6,color_discrete_map={'정상':'#4b79b8','이상':'#e06464'}),width="stretch",key='distribution')
    with tabs[3]:
        st.subheader('관리자 점검 메모')
        if selected is None: st.info('이상 사건이 생기면 사건별 점검 메모를 남길 수 있습니다.')
        else:
            event_id=selected.name
            st.write(f'{selected.equipment_id} · {selected.start}')
            db=BASE/'storage/notes.sqlite3'
            with st.form('memo_form_'+str(event_id)):
                states=['미확인','확인 중','현장 점검 필요','조치 완료']
                state=st.selectbox('사건 처리 상태',states,index=states.index(event_state(db,fingerprint,event_id)),key='state_'+event_id)
                memo=st.text_area('점검 내용',value=read_note(db,fingerprint,event_id),key='memo_'+event_id)
                if st.form_submit_button('메모 저장'):
                    try:
                        save_note(db,fingerprint,event_id,memo)
                        event_state(db,fingerprint,event_id,state)
                        st.success('저장했습니다.')
                    except Exception as e: st.error(f'메모 저장 실패: {e}')


    with tabs[4]:
        st.subheader('지속 이상 분석')
        st.badge('규칙 기반 · AI 미연결', color='gray')
        if selected is None:
            st.info('이상 사건이 생기면 상세 탭에서 사건을 선택하세요.')
        else:
            eid=selected.name
            st.write(f'{selected.equipment_id} · {selected.start}')
            if st.button('선택 사건 지금 분석',key='manual_analysis'):
                result=analyze(selected,data,shap,horizon,float(cadence))
                result['trigger']='관리자 수동 요청';result['updates']=ss.reports.get(eid,{}).get('updates',0)+1
                ss.reports[eid]=result
            report=ss.reports.get(eid)
            if report:
                render_report(report)
                with st.expander('분석 이력'):
                    st.write(f"갱신 {report['updates']}회 · {report['trigger']}")
                    for tool in report['trace']: st.write('✓ '+tool)
                st.download_button('분석 요약 다운로드',report['text'],file_name='OHT_analysis.txt')
            else: st.info(f'분석 대기 · {analysis_wait:g}초 지속 시 실행')
            st.subheader('선택 사건에 질문')
            question=st.selectbox('확인할 내용',['왜 이 센서를 확인해야 해?','이전에도 이런 사건이 있었어?','점검 순위와 전달 내용을 정리해줘.'])
            if st.button('근거 조회',key='question_lookup'):
                answer=analyze(selected,data,shap,horizon,float(cadence))
                st.markdown(f'**{question}**')
                render_report(answer, question)
            with st.expander('센서·파생변수 정의'):
                st.dataframe(pd.DataFrame(MEANINGS.items(),columns=['센서','측정 대상']),hide_index=True,width='stretch')
                st.write('Diff: 동일 세션 직전 관측과의 차이. MA_5·MA_30: 현재 포함 최근 5개·30개 관측 평균. 세션: 관측 간격 5초 초과 시 분리. CT_Std는 고장 또는 전류 불균형의 확정 지표가 아닙니다.')


dashboard()
