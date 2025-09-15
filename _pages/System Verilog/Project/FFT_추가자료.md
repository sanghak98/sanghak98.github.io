---
title: "FFT_추가자료"
date: "2025-08-04"
thumbnail: "/assets/img/thumbnail/sv.jpg"
---

# ✅ Setup Time Slack 이슈와 해결 방법

## 1. 개념 정리

### 📌 Slack이란?

```
Slack = 클럭 주기 - (Data arrival time - Clock skew)
```

- **Slack < 0** → Setup timing violation 발생
- **Slack ≥ 0** → 타이밍 조건 충족 (met)

---

## 2. Slack (Setup Time Violation) 원인 요약

| 원인 | 설명 |
|------|------|
| 논리 경로가 너무 길다 | 조합 로직이 깊어서 data arrival time이 길어짐 |
| Fan-out이 많다 | 하나의 신호가 여러 로직을 구동해서 지연 발생 |
| Gate 지연이 크다 | Slow cell 사용, 복잡한 연산 등 |
| Clock Skew가 크다 | Clock 도달 시간이 레지스터마다 다름 |
| Wire delay가 큼 | 배치 및 배선 단계에서 경로 길이가 너무 길어짐 |
| Setup time 자체가 큼 | 플립플롭의 특성상 필요한 setup time이 큼 |

---

## 3. Setup Slack을 줄이는 (혹은 메우는) 방법

### ① Critical Path 단축 (로직 최적화)
- 조합 논리가 너무 길면 분해 (분기, 파이프라인)
- 불필요한 논리 제거 (`logic optimization`)
- shift, multiply 등의 무거운 연산은 미리 계산하거나 LUT로 대체

---

### ② 파이프라인 삽입
- 긴 조합 경로를 나눠서 **레지스터 중간 삽입**
- 클럭 하나에 처리할 양을 줄이고, throughput을 올림

📌 *단점: latency 증가, FSM이 복잡해질 수 있음*

---

### ③ High Drive Cell 사용
- 지연이 큰 경로에 고속 셀, 고드라이브 셀을 배치  
- 예: `INV_X1` → `INV_X4`  

---

### ④ Gate Sizing (셀 크기 조정)
- 드라이버의 드라이브 강도를 증가시켜 delay 감소  
- 너무 크면 전력, 면적 증가 → 균형 필요

---

### ⑤ 경로 분할 (Re-structuring)
- 복잡한 path를 여러 경로로 분기해서 병렬 처리  
- 예: 큰 decoder를 계층 구조로 분리

---

### ⑥ Clock Skew 최소화
- CTS 단계에서 clock 도달 시간 균일화

---

### ⑦ 클럭 도메인 분리 / 클럭 속도 줄이기
- 타이밍이 안 맞는다면 해당 블록만 slower clock 사용
- *주의: CDC 처리 필요*

---

### ⑧ Physical Optimization (배치/배선)
- P&R 단계에서 경로를 짧게 하도록 제약 설정  
- Net Delay 줄이기

---

## 4. Tool 상에서 하는 조치 (Synthesis/Vivado 기준)

- `set_max_delay` 제약 재조정
- `compile_ultra`, `optimize_registers`, `retime` 옵션 사용
- `report_timing` 분석 후 critical path 타겟팅

```tcl
# Vivado에서 타이밍 경로 분석
report_timing -sort_by group -max_paths 10
```

---

## 5. 결론: 어떻게 slack 문제를 해결할까?

| 방법 | 요약 |
|------|------|
| 조합 경로 줄이기 | 로직 단순화, 파이프라인, 병렬화 |
| 셀 크기 조절 | 고속 셀/고드라이브 셀로 변경 |
| 물리적 최적화 | 배치/배선 경로 단축, 레이아웃 조정 |
| 툴 옵션 활용 | 자동 retiming, compile_ultra |
| 클럭 전략 | clock skew 조절, slower clock 분리 |