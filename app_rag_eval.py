# rag_eval.py
# 제조현장 RAG 평가 스크립트 - 진행률/소요시간/중단 재개 지원 버전
#
# 실행 예시:
#   python rag_eval.py --eval_csv eval_questions.csv --output_csv rag_eval_results.csv
#   python rag_eval.py --eval_csv eval_questions.csv --output_csv rag_eval_results.csv --resume
#   python rag_eval.py --eval_csv eval_questions.csv --output_csv rag_eval_results.csv --only_rag
#
# 필요 패키지:
#   pip install pandas rouge-score nltk bert-score
#
# 선택 패키지:
#   pip install tqdm
#
# 실행 전 확인:
#   1) Ollama 실행: ollama serve
#   2) 모델 확인: ollama list
#   3) Chroma DB 생성 완료: ./chroma_db 폴더 존재

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except Exception:
    BERT_SCORE_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# =========================
# 기본 설정
# =========================

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
LLM_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_K = 10

UNKNOWN_PATTERNS = [
    "확인할 수 없습니다",
    "제공된 문서",
    "문서에서 확인",
    "알 수 없습니다",
    "근거가 없습니다",
    "찾을 수 없습니다",
]


RAG_PROMPT = PromptTemplate.from_template(
    """
당신은 제조 현장 문서 기반 RAG 질의응답 assistant입니다.
반드시 아래 문맥에 근거해서만 답변하세요.

규칙:
1. 문맥에 없는 내용은 추측하지 말고 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.
2. 답변에는 핵심 근거를 간단히 포함하세요.
3. 여러 문서 내용이 충돌하면 충돌한다고 명시하세요.
4. 답변은 제조 현장 작업자가 이해하기 쉽게 간결하게 작성하세요.

문맥:
{context}

질문:
{question}

답변:
"""
)

NO_RAG_PROMPT = PromptTemplate.from_template(
    """
다음 질문에 답하세요.
문서 검색 없이 모델 자체 지식만 사용합니다.
확실하지 않은 내용은 추측하지 말고 알 수 없다고 답하세요.

질문:
{question}

답변:
"""
)

JUDGE_PROMPT = """
당신은 RAG 질의응답 평가자입니다.
아래 질문, 기준 정답, 모델 답변을 보고 1~10점으로 평가하세요.

평가 기준:
- usefulness: 사용자가 실제로 활용할 수 있는가?
- relevance: 질문과 관련 있는가?
- conciseness: 불필요하게 장황하지 않은가?
- faithfulness: 기준 정답/문서 근거 범위를 벗어나지 않는가?

반드시 JSON만 출력하세요.

질문:
{question}

기준 정답:
{gold_answer}

모델 답변:
{answer}

출력 형식:
{{
  "usefulness": 0,
  "relevance": 0,
  "conciseness": 0,
  "faithfulness": 0,
  "reason": "짧은 평가 이유"
}}
"""


# =========================
# 유틸
# =========================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sec_to_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}시간 {m}분 {s}초"
    if m:
        return f"{m}분 {s}초"
    return f"{s}초"


def approx_token_count(text: str) -> int:
    """
    Ollama는 OpenAI API처럼 usage token을 바로 주지 않는 경우가 많아
    평가군 간 상대 비교용 근사값을 사용한다.
    """
    if not text:
        return 0
    return max(1, len(str(text)) // 2)


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def normalize_source_list(source_text: str) -> List[str]:
    if not source_text or str(source_text).lower() == "nan":
        return []
    return [s.strip() for s in str(source_text).split("|") if s.strip()]


def append_row_csv(output_csv: str, row: Dict):
    """
    문항 하나 끝날 때마다 즉시 저장한다.
    중간에 멈춰도 이미 완료된 결과는 CSV에 남는다.
    """
    out_path = Path(output_csv)
    df = pd.DataFrame([row])
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False, encoding="utf-8-sig")


def save_checkpoint(checkpoint_csv: str, row: Dict):
    """
    마지막 진행 문항 확인용 별도 체크포인트 파일.
    """
    out_path = Path(checkpoint_csv)
    row2 = dict(row)
    row2["checkpoint_time"] = now_str()
    df = pd.DataFrame([row2])
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False, encoding="utf-8-sig")


def load_completed_keys(output_csv: str) -> set:
    """
    resume 모드에서 이미 완료된 (id, mode)를 건너뛰기 위한 키 목록.
    """
    out_path = Path(output_csv)
    if not out_path.exists():
        return set()

    try:
        df = pd.read_csv(out_path)
        if "id" not in df.columns or "mode" not in df.columns:
            return set()
        return set(zip(df["id"].astype(str), df["mode"].astype(str)))
    except Exception:
        return set()


# =========================
# 모델 / DB 로드
# =========================

def check_ollama_connection(base_url: str) -> bool:
    """
    requests를 강제 의존하지 않기 위해 httpx를 사용하지 않고 urllib로 간단 확인.
    """
    import urllib.request
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def load_vectorstore(persist_dir: str, embed_model: str):
    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"벡터DB 폴더를 찾을 수 없습니다: {persist_dir}\n"
            "먼저 app_ollama.py에서 PDF를 업로드하고 DB를 생성하세요."
        )

    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )


def load_llm(model: str, base_url: str):
    return OllamaLLM(
        model=model,
        temperature=0.1,
        base_url=base_url,
        num_predict=256,
        num_ctx=4096,
        keep_alive="10m"
    )


# =========================
# 답변 생성
# =========================

def run_rag(question: str, db, llm, k: int = DEFAULT_K) -> Dict:
    started_at = time.time()

    docs_with_scores = db.similarity_search_with_score(question, k=k)

    contexts = []
    sources = []
    scores = []

    for doc, score in docs_with_scores:
        contexts.append(doc.page_content)
        sources.append(doc.metadata.get("source", "unknown"))
        scores.append(safe_float(score))

    context_text = "\n\n".join(contexts)

    final_prompt = RAG_PROMPT.format(
        context=context_text,
        question=question
    )

    answer = llm.invoke(final_prompt)
    elapsed = time.time() - started_at

    return {
        "mode": "RAG",
        "answer": answer,
        "sources": "|".join(sources),
        "scores": "|".join("" if s is None else str(s) for s in scores),
        "top1_source": sources[0] if sources else "",
        "top1_score": scores[0] if scores else None,
        "retrieved_count": len(sources),
        "response_time_sec": round(elapsed, 3),
        "answer_token_approx": approx_token_count(answer),
    }


def run_no_rag(question: str, llm) -> Dict:
    started_at = time.time()

    final_prompt = NO_RAG_PROMPT.format(question=question)
    answer = llm.invoke(final_prompt)

    elapsed = time.time() - started_at

    return {
        "mode": "NO_RAG",
        "answer": answer,
        "sources": "",
        "scores": "",
        "top1_source": "",
        "top1_score": None,
        "retrieved_count": 0,
        "response_time_sec": round(elapsed, 3),
        "answer_token_approx": approx_token_count(answer),
    }


# =========================
# 평가 지표
# =========================

def calc_rouge_l(reference: str, prediction: str) -> float:
    reference = str(reference or "")
    prediction = str(prediction or "")

    if not reference.strip() or not prediction.strip():
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 6)


def calc_bleu(reference: str, prediction: str) -> float:
    """
    한국어 형태소 분석기 없이 실행 가능한 문자 단위 BLEU.
    실험군 간 상대 비교용으로 사용.
    """
    reference = str(reference or "")
    prediction = str(prediction or "")

    ref_tokens = list(reference.replace(" ", ""))
    pred_tokens = list(prediction.replace(" ", ""))

    if not ref_tokens or not pred_tokens:
        return 0.0

    smoothie = SmoothingFunction().method1
    value = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    return round(float(value), 6)


def calc_source_hit(pred_sources: str, gold_source: str) -> int:
    gold_sources = normalize_source_list(gold_source)
    if not gold_sources:
        return 0

    pred_sources_text = str(pred_sources or "")

    return int(any(gs in pred_sources_text for gs in gold_sources))


def calc_unknown_correct(answer: str, answerable: str) -> int:
    if str(answerable).strip().lower() != "no":
        return 0

    answer = str(answer or "")
    return int(any(pattern in answer for pattern in UNKNOWN_PATTERNS))


def llm_as_judge(question: str, gold_answer: str, answer: str, llm) -> Dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        answer=answer
    )

    raw = llm.invoke(prompt)

    # 모델이 JSON 앞뒤로 잡문을 붙이는 경우를 대비해 중괄호 구간만 추출 시도
    raw_text = str(raw).strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start != -1 and end != -1 and end > start:
        raw_json = raw_text[start:end + 1]
    else:
        raw_json = raw_text

    try:
        parsed = json.loads(raw_json)
    except Exception:
        parsed = {
            "usefulness": None,
            "relevance": None,
            "conciseness": None,
            "faithfulness": None,
            "reason": raw_text[:500],
        }

    return {
        "judge_usefulness": parsed.get("usefulness"),
        "judge_relevance": parsed.get("relevance"),
        "judge_conciseness": parsed.get("conciseness"),
        "judge_faithfulness": parsed.get("faithfulness"),
        "judge_reason": parsed.get("reason"),
    }


def add_basic_metrics(row: Dict) -> Dict:
    row["rouge_l"] = calc_rouge_l(row.get("gold_answer", ""), row.get("answer", ""))
    row["bleu"] = calc_bleu(row.get("gold_answer", ""), row.get("answer", ""))
    row["source_hit"] = calc_source_hit(row.get("sources", ""), row.get("gold_source", ""))
    row["unknown_correct"] = calc_unknown_correct(row.get("answer", ""), row.get("answerable", "yes"))
    return row


def add_bert_score_to_csv(output_csv: str):
    """
    모든 결과를 저장한 뒤 BERTScore를 한 번에 계산한다.
    중간중간 계산하면 너무 느려서 마지막에 일괄 처리.
    """
    if not BERT_SCORE_AVAILABLE:
        print("[안내] bert-score 패키지를 사용할 수 없어 BERTScore 계산은 건너뜁니다.")
        return

    out_path = Path(output_csv)
    if not out_path.exists():
        return

    df = pd.read_csv(out_path)
    if df.empty:
        return

    if "bertscore_f1" in df.columns and df["bertscore_f1"].notna().all():
        print("[안내] 기존 결과에 BERTScore가 이미 있어 건너뜁니다.")
        return

    print(f"[{now_str()}] BERTScore 계산 시작: {len(df)}개 답변")
    preds = df["answer"].fillna("").astype(str).tolist()
    refs = df["gold_answer"].fillna("").astype(str).tolist()

    try:
        _, _, f1 = bert_score(preds, refs, lang="ko", rescale_with_baseline=False)
        df["bertscore_f1"] = [round(float(x), 6) for x in f1]
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[{now_str()}] BERTScore 계산 완료")
    except Exception:
        print("[경고] BERTScore 계산 중 오류가 발생했습니다.")
        traceback.print_exc()


# =========================
# 진행률 출력
# =========================

def print_progress(
    current_index: int,
    total_tasks: int,
    qid: str,
    mode: str,
    question: str,
    global_start_time: float,
    item_start_time: Optional[float] = None,
    status: str = "START"
):
    elapsed_total = time.time() - global_start_time
    avg_per_task = elapsed_total / max(1, current_index - 1) if current_index > 1 else 0
    remain_tasks = total_tasks - current_index + 1
    eta = avg_per_task * remain_tasks if avg_per_task else 0
    percent = current_index / max(1, total_tasks) * 100

    short_question = question.replace("\n", " ")
    if len(short_question) > 80:
        short_question = short_question[:80] + "..."

    if item_start_time and status == "END":
        item_elapsed = time.time() - item_start_time
        item_text = f" | 문항 소요: {sec_to_hms(item_elapsed)}"
    else:
        item_text = ""

    print(
        f"[{now_str()}] [{status}] "
        f"{current_index}/{total_tasks} ({percent:.1f}%) | "
        f"ID={qid} | MODE={mode} | "
        f"전체 경과={sec_to_hms(elapsed_total)} | 예상 남은 시간={sec_to_hms(eta)}"
        f"{item_text}\n"
        f"  질문: {short_question}",
        flush=True
    )


# =========================
# 메인 평가 루프
# =========================

def build_tasks(df: pd.DataFrame, only_rag: bool, only_no_rag: bool) -> List[Tuple[int, str]]:
    modes = []
    if only_rag:
        modes = ["RAG"]
    elif only_no_rag:
        modes = ["NO_RAG"]
    else:
        modes = ["NO_RAG", "RAG"]

    tasks = []
    for idx in range(len(df)):
        for mode in modes:
            tasks.append((idx, mode))
    return tasks


def evaluate(args):
    eval_csv = args.eval_csv
    output_csv = args.output_csv
    checkpoint_csv = args.checkpoint_csv

    if not Path(eval_csv).exists():
        raise FileNotFoundError(f"평가 CSV를 찾을 수 없습니다: {eval_csv}")

    df = pd.read_csv(eval_csv)

    required_cols = ["id", "question", "gold_answer", "gold_source", "answerable"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"평가 CSV에 필요한 컬럼이 없습니다: {missing_cols}")

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    print("=" * 80)
    print("제조현장 RAG 평가 시작")
    print("=" * 80)
    print(f"시작 시간       : {now_str()}")
    print(f"평가 파일       : {eval_csv}")
    print(f"결과 파일       : {output_csv}")
    print(f"체크포인트 파일 : {checkpoint_csv}")
    print(f"문항 수         : {len(df)}")
    print(f"모델            : {args.llm_model}")
    print(f"임베딩 모델     : {args.embed_model}")
    print(f"벡터DB          : {args.persist_dir}")
    print(f"검색 k          : {args.k}")
    print(f"resume          : {args.resume}")
    print(f"only_rag        : {args.only_rag}")
    print(f"only_no_rag     : {args.only_no_rag}")
    print(f"judge           : {args.judge}")
    print(f"bertscore       : {args.bertscore}")
    print("=" * 80, flush=True)

    if not check_ollama_connection(args.ollama_base_url):
        raise ConnectionError(
            f"Ollama 서버에 연결할 수 없습니다: {args.ollama_base_url}\n"
            "PowerShell 또는 CMD에서 'ollama serve'를 실행한 뒤 다시 시도하세요."
        )

    db = None
    if not args.only_no_rag:
        print(f"[{now_str()}] 벡터DB 로드 중...")
        db = load_vectorstore(args.persist_dir, args.embed_model)
        try:
            print(f"[{now_str()}] 벡터DB 로드 완료. 청크 수: {db._collection.count()}")
        except Exception:
            print(f"[{now_str()}] 벡터DB 로드 완료.")

    print(f"[{now_str()}] Ollama LLM 로드 중...")
    llm = load_llm(args.llm_model, args.ollama_base_url)
    print(f"[{now_str()}] Ollama LLM 로드 완료.")

    completed_keys = load_completed_keys(output_csv) if args.resume else set()

    tasks = build_tasks(df, args.only_rag, args.only_no_rag)
    total_tasks = len(tasks)
    global_start_time = time.time()

    if TQDM_AVAILABLE and not args.no_tqdm:
        iterator = tqdm(enumerate(tasks, start=1), total=total_tasks, desc="Evaluating", unit="task")
    else:
        iterator = enumerate(tasks, start=1)

    for task_no, (row_idx, mode) in iterator:
        row_src = df.iloc[row_idx]
        qid = str(row_src["id"])
        question = str(row_src["question"])

        if (qid, mode) in completed_keys:
            if not (TQDM_AVAILABLE and not args.no_tqdm):
                print(f"[{now_str()}] [SKIP] {task_no}/{total_tasks} | ID={qid} | MODE={mode} 이미 완료됨", flush=True)
            continue

        print_progress(
            current_index=task_no,
            total_tasks=total_tasks,
            qid=qid,
            mode=mode,
            question=question,
            global_start_time=global_start_time,
            status="START"
        )

        item_start_time = time.time()

        base = {
            "id": qid,
            "type": row_src.get("type", ""),
            "question": question,
            "gold_answer": str(row_src.get("gold_answer", "")),
            "gold_source": str(row_src.get("gold_source", "")) if not pd.isna(row_src.get("gold_source", "")) else "",
            "answerable": str(row_src.get("answerable", "yes")),
            "evaluation_focus": str(row_src.get("evaluation_focus", "")),
            "source_url": str(row_src.get("source_url", "")) if not pd.isna(row_src.get("source_url", "")) else "",
            "started_at": now_str(),
        }

        try:
            if mode == "NO_RAG":
                result = run_no_rag(question, llm)
            elif mode == "RAG":
                if db is None:
                    raise RuntimeError("RAG 모드인데 벡터DB가 로드되지 않았습니다.")
                result = run_rag(question, db, llm, k=args.k)
            else:
                raise ValueError(f"알 수 없는 mode: {mode}")

            final_row = {**base, **result}
            final_row = add_basic_metrics(final_row)

            if args.judge:
                judge_result = llm_as_judge(
                    question=question,
                    gold_answer=final_row["gold_answer"],
                    answer=final_row["answer"],
                    llm=llm
                )
                final_row.update(judge_result)

            final_row["finished_at"] = now_str()
            final_row["status"] = "success"
            final_row["error"] = ""

        except Exception as e:
            final_row = {
                **base,
                "mode": mode,
                "answer": "",
                "sources": "",
                "scores": "",
                "top1_source": "",
                "top1_score": None,
                "retrieved_count": 0,
                "response_time_sec": round(time.time() - item_start_time, 3),
                "answer_token_approx": 0,
                "rouge_l": 0.0,
                "bleu": 0.0,
                "source_hit": 0,
                "unknown_correct": 0,
                "finished_at": now_str(),
                "status": "error",
                "error": traceback.format_exc(),
            }
            print("[오류] 문항 처리 중 오류 발생")
            print(final_row["error"], flush=True)

        append_row_csv(output_csv, final_row)
        save_checkpoint(checkpoint_csv, final_row)

        print_progress(
            current_index=task_no,
            total_tasks=total_tasks,
            qid=qid,
            mode=mode,
            question=question,
            global_start_time=global_start_time,
            item_start_time=item_start_time,
            status="END"
        )

    if args.bertscore:
        add_bert_score_to_csv(output_csv)

    print_summary(output_csv)

    print("=" * 80)
    print(f"평가 완료: {now_str()}")
    print(f"전체 소요: {sec_to_hms(time.time() - global_start_time)}")
    print(f"결과 파일: {output_csv}")
    print("=" * 80)


def print_summary(output_csv: str):
    out_path = Path(output_csv)
    if not out_path.exists():
        return

    try:
        df = pd.read_csv(out_path)
        if df.empty:
            return

        print("\n" + "=" * 80)
        print("평가 요약")
        print("=" * 80)

        metrics = ["rouge_l", "bleu", "source_hit", "unknown_correct", "response_time_sec", "answer_token_approx"]
        if "bertscore_f1" in df.columns:
            metrics.append("bertscore_f1")

        available_metrics = [m for m in metrics if m in df.columns]

        summary = df.groupby("mode")[available_metrics].agg(["mean", "std", "count"])
        print(summary)

        if "status" in df.columns:
            print("\n상태별 개수")
            print(df["status"].value_counts())

        print("=" * 80 + "\n")
    except Exception:
        print("[경고] 요약 출력 중 오류 발생")
        traceback.print_exc()


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="제조현장 RAG 평가 스크립트")

    parser.add_argument("--eval_csv", default="eval_questions.csv", help="평가 질문 CSV 경로")
    parser.add_argument("--output_csv", default="rag_eval_results.csv", help="평가 결과 CSV 경로")
    parser.add_argument("--checkpoint_csv", default="rag_eval_checkpoint.csv", help="진행 체크포인트 CSV 경로")

    parser.add_argument("--persist_dir", default=PERSIST_DIR, help="Chroma DB persist directory")
    parser.add_argument("--embed_model", default=EMBED_MODEL, help="HuggingFace embedding model")
    parser.add_argument("--llm_model", default=LLM_MODEL, help="Ollama model name")
    parser.add_argument("--ollama_base_url", default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="검색 문서 수 k")

    parser.add_argument("--resume", action="store_true", help="이미 완료된 id/mode 결과는 건너뜀")
    parser.add_argument("--only_rag", action="store_true", help="RAG만 평가")
    parser.add_argument("--only_no_rag", action="store_true", help="NO_RAG만 평가")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개 문항만 테스트")

    parser.add_argument("--judge", action="store_true", help="LLM-as-Judge 평가 추가")
    parser.add_argument("--bertscore", action="store_true", help="평가 종료 후 BERTScore 계산")
    parser.add_argument("--no_tqdm", action="store_true", help="tqdm 진행바 사용 안 함")

    args = parser.parse_args()

    if args.only_rag and args.only_no_rag:
        parser.error("--only_rag와 --only_no_rag는 동시에 사용할 수 없습니다.")

    return args


if __name__ == "__main__":
    try:
        evaluate(parse_args())
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 평가를 중단했습니다.")
        print("이미 완료된 문항은 output_csv에 저장되어 있습니다.")
        print("다음 실행 시 --resume 옵션을 사용하면 완료된 문항을 건너뜁니다.")
        sys.exit(130)
    except Exception:
        print("\n[치명적 오류] 평가 실행 실패")
        traceback.print_exc()
        sys.exit(1)
