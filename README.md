# OHT 관제 앱 온라인 배포 안내

## 준비된 파일

app.py가 실행 파일입니다. data/replay.csv와 data/shap.csv를 읽습니다.
기존 코드가 앱 파일 위치를 기준으로 경로를 계산하므로 Windows 전용 경로 수정은 필요하지 않았습니다.
requirements.txt도 이미 포함되어 있습니다. 학습 모델을 다시 학습하지 않고 저장된 예측과 SHAP으로 시연합니다.

## GitHub에 업로드

1. https://github.com 에 로그인하고 New repository를 선택합니다.
2. 저장소 이름을 oht-monitoring으로 정합니다. 코드와 데이터의 열람 범위를 제한하려면 Private을 선택합니다.
3. 이 ZIP을 압축 해제합니다.
4. 저장소에서 Add file → Upload files를 선택합니다.
5. OHT_Monitor 폴더 안의 파일과 data, tests 폴더를 올립니다. ZIP 자체를 올리는 방식으로는 실행되지 않습니다.
6. storage 폴더는 기존 관리자 메모입니다. 기존 메모를 온라인에 복사하려는 경우에만 별도로 올립니다. 업로드하지 않아도 앱이 새 저장 파일을 자동으로 만듭니다.
7. Commit changes를 누릅니다.

저장소 첫 화면에 app.py와 requirements.txt가 보이고 data 안에 CSV가 있는지 확인합니다.
.gitignore는 Git으로 업로드할 때 로컬 메모 파일을 제외합니다. 브라우저에서 수동 업로드할 때도 storage를 직접 제외해야 합니다.

## Streamlit에서 배포

1. https://share.streamlit.io 에서 GitHub 계정으로 로그인합니다.
2. 비공개 저장소라면 해당 저장소에 대한 연결 권한도 설정합니다.
3. Create app → Yup, I have an app을 선택합니다.
4. Repository에 본인계정/oht-monitoring을 선택합니다.
5. Branch는 main, Main file path는 app.py로 설정합니다.
6. Advanced settings에서 Python 3.12를 선택합니다.
7. Deploy를 누르고 설치와 실행이 끝날 때까지 기다립니다.
8. 생성된 streamlit.app 주소를 엽니다.

OHT_Monitor 폴더 자체를 올려 저장소 안에 app.py가 하위 폴더로 들어갔다면 Main file path를 OHT_Monitor/app.py로 바꿉니다. requirements.txt와 data는 app.py와 같은 폴더 안에 유지합니다.

## 교수님과 공유

앱의 공유 설정을 확인합니다. 로그인 없이 열게 하려면 앱을 공개로 설정합니다.
교수님에게만 열람을 허용하려면 앱을 비공개로 두고 교수님 이메일에 권한을 줍니다.
각 접속자는 자신의 재생 시각과 필터를 조작합니다. 발표자와 관람자의 재생 화면이 자동으로 동기화되지는 않습니다.
앱의 실시간 관제 모드는 기존 CSV를 시간 순서대로 재생하는 시연입니다.

## 관리자 메모와 처리 상태

메모와 처리 상태는 storage/notes.sqlite3에 저장됩니다.
Community Cloud의 로컬 저장은 영구 보관이 보장되지 않습니다. 재배포나 실행 환경 교체 등으로 새로 쓴 메모가 사라질 수 있습니다.
이번 배포 패키지는 기존 SQLite 방식을 유지합니다. 외부 DB 연결은 포함하지 않았습니다.
온라인에서 쓴 메모는 다영님 PC의 파일에 자동으로 반영되지 않습니다.
같은 배포 앱의 사용자들은 서버의 메모와 처리 상태를 공유하며 현재 앱에는 관리자 인증 기능이 없습니다. 앱에 접근할 수 있는 사람은 메모와 상태를 변경할 수 있습니다.
장기간 보관이 필요하면 외부 DB 연결을 추가해야 합니다.

## 배포 후 확인

- 첫 화면이 열리는지 확인합니다.
- 다음 관측 시각 버튼으로 센서가 표시되는지 확인합니다.
- 날짜별 조회에서 장비와 이상 사건을 선택합니다.
- XAI 근거와 선택 사건 지금 분석 기능을 확인합니다.
- 테스트 메모를 저장하고 같은 사건을 다시 선택해 확인합니다.

오류가 뜨면 Streamlit의 Manage app에서 로그를 확인합니다.
ModuleNotFoundError는 패키지 목록, FileNotFoundError는 업로드 파일과 대소문자를 확인합니다.

공식 문서

https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

## 이번 수정

수동 분석에서 사건 ID가 인덱스로 이동해 분석이 실패하는 오류를 수정했습니다. 선택 목록을 만들 때 사건 ID 열을 유지하도록 변경했습니다.
