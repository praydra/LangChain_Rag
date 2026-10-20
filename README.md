# LangChain 기반 제조 안전 문서 RAG 시스템

한국산업안전보건공단(KOSHA)의 제조업 안전 문서를 기반으로 질문에 답하는 **로컬 RAG(Retrieval-Augmented Generation) 시스템**입니다.
RAG를 적용했을 때와 LLM 단독(NO_RAG)일 때의 답변 품질을 정량적으로 비교했습니다.

## 데이터셋

| 경로 | 내용 |
| --- | --- |
| `SampleDoc/` | KOSHA 제조업 안전 PDF 5종(위험성평가 지침, 끼임사고 안전작업절차·예방 제언, 유해위험방지계획서 안내, 보호구 사용법) |
| `eval_questions.csv` | 평가 질문 66개와 정답, 정답 출처 문서 |
| `chroma_db/` | 위 문서로 구축한 Chroma 벡터 DB |

평가 질문 유형은 simple 34개, condition 20개, unknown(문서에 답이 없는 질문) 6개, multi 4개, summary 2개입니다.

## 소스 구성

```text
LangChain_Rag/
├── app_ollama.py           # Streamlit 문서 업로드·벡터 DB 구축·질의응답 앱
├── app_rag_eval.py         # RAG vs NO_RAG 정량 평가 스크립트
├── rag_eval_results.csv    # 평가 결과
└── rag_eval_checkpoint.csv # 평가 진행 체크포인트
```

| 구성 요소 | 사용 기술 |
| --- | --- |
| UI | Streamlit |
| 문서 로딩 | PDF(pypdf, 스캔본은 Tesseract OCR), Word, CSV, 이미지 OCR |
| 청킹 | `RecursiveCharacterTextSplitter` (chunk 1000, overlap 200) |
| 임베딩 | `jhgan/ko-sroberta-multitask` (한국어 특화) |
| 벡터 DB | Chroma |
| LLM | Ollama `gpt-oss:20b` (로컬) |

## 분석 내용

- `app_ollama.py`는 여러 문서를 업로드받아 텍스트를 추출하고, 청킹·임베딩한 뒤 Chroma에 저장합니다. 스캔본 PDF는 자동으로 감지해 OCR을 수행합니다.
- `app_rag_eval.py`는 같은 질문을 두 방식으로 답변시키고 비교합니다.
  - **RAG**: 상위 k=10개 청크를 검색한 뒤 답변을 생성합니다.
  - **NO_RAG**: LLM만으로 답변을 생성합니다.
- 평가 지표: ROUGE-L, BLEU, 출처 적중률(source hit), 답변 불가 질문 처리(unknown correct), 응답 시간
- 선택 옵션: LLM-as-Judge(`--judge`), BERTScore(`--bertscore`). 체크포인트 기반으로 이어서 실행할 수 있습니다(`--resume`).

## 결과 (`rag_eval_results.csv`, 질문 66개)

| 모드 | 출처 적중률 | ROUGE-L | BLEU | 평균 응답 시간 |
| --- | --- | --- | --- | --- |
| RAG (k=10) | 50.0% | 0.006 | 0.022 | 13.2초 |
| NO_RAG | 0.0% | 0.040 | 0.047 | 6.5초 |

- RAG는 질문의 50%에서 정답 근거 문서를 검색했습니다.
- ROUGE-L과 BLEU는 두 방식 모두 낮았습니다. 생성형 답변의 표현이 정답 문장과 달라 단어 일치 기반 지표로는 차이가 잘 드러나지 않습니다.

## 실행 방법

```bash
pip install streamlit langchain-classic langchain-community langchain-chroma langchain-huggingface langchain-ollama langchain-text-splitters pypdf pdf2image pytesseract pillow docx2txt pandas rouge-score nltk
ollama pull gpt-oss:20b
streamlit run app_ollama.py
python app_rag_eval.py --eval_csv eval_questions.csv --k 10 --resume
```

- 벡터 DB(`./chroma_db`)와 평가 CSV는 실행 위치 기준 상대 경로이므로, `LangChain_Rag` 폴더에서 실행하세요.
- 스캔본 OCR에는 Tesseract와 Poppler가 필요합니다. Tesseract가 PATH에 없다면 환경변수 `TESSERACT_CMD`로 실행 파일 경로를 지정하세요.

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```
