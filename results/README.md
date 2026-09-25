# 결과 단일 출처

이 디렉터리는 `/mnt/synology_nas_00/chanwcom/results/` 의 사본입니다. 세
머신이 NAS 를 공유하므로 **생성·갱신은 NAS 쪽 경로에서 하고**, 여기에는
커밋 시점의 스냅샷이 들어옵니다.

    rsync -a /mnt/synology_nas_00/chanwcom/results/ results/

`--delete` 를 붙이지 마십시오 -- 이 `README.md` 는 repo 쪽에만 있어서 같이
지워집니다. NAS 에서 없어진 파일이 여기 남아 표를 오도할 것 같으면 지우는
대신 `_superseded` 스텁으로 덮고 원본 경로를 적어 두십시오
(`ctc_test/results.jsonl` 이 그 예입니다).

| 파일 | 무엇 |
|---|---|
| `RESULTS.md` | 검증된 test 결과. `wav2vec2_inference.py` / `rnnt/eval_ckpt.py` 의 greedy 출력만, test-clean·test-other 전체만 |
| `CHECKPOINTS.md` | 어느 체크포인트가 어디 있나 + **방법 이름 대응표** |
| `verified.jsonl` | 셀 단위 원본. 줄마다 `run`·`ckpt` 경로 포함 |
| `checkpoints.jsonl` | 체크포인트 전수 (CTC 460 + RNN-T 358) |
| `rnnt_test/`, `ctc_test/`, `slimducer_test/` | 평가 원본. 줄마다 `how`(스크립트·디코더·dtype·batch·commit) |
| `EVAL_PROTOCOL.md` | **측정 규약.** 어떤 숫자가 표에 들어올 수 있는지와 그 실측 근거 |
| `bs1_test/results.jsonl` | 현재 유효한 CTC·RNN-T 원본 (batch_size=1) |
| `DISCARDED_bs8/` | batch_size=8 시절 측정값. 보존용이며 표에 쓰지 않습니다 |
| `deleted_*.json/.txt` | 삭제 매니페스트 |
| `running_argv_20260924.jsonl` | 도는 잡의 전체 argv (CTC 가 인자를 저장하지 않던 시절의 보존본) |

**먼저 읽을 것**: 전통 LS 는 `fas` 라는 이름의 디렉터리에 들어 있습니다
(FAS + beta=1.0 이 textbook LS 와 bit-exact). `CHECKPOINTS.md` 의 대응표를
보지 않고 디렉터리 이름만 믿으면 표를 잘못 읽습니다.
