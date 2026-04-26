from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

from benchmark_hippo5_softbox_retrieval import StudentEncoder


DEFAULT_LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_TOP_K = [1, 3, 5, 10, 20]


GO_BENCH_SOURCE = r'''
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"hippo5/database"
)

type Conversation struct {
	ID        string     `json:"id"`
	Records   []Record   `json:"records"`
	Questions []Question `json:"questions"`
}

type Record struct {
	DiaID  string    `json:"dia_id"`
	Text   string    `json:"text"`
	Vector []float32 `json:"vector"`
}

type Question struct {
	ID       string    `json:"id"`
	Category int      `json:"category"`
	Question string   `json:"question"`
	Evidence []string `json:"evidence"`
	Vector   []float32 `json:"vector"`
}

type Metric struct {
	HitRate        float64 `json:"hit_rate"`
	EvidenceRecall float64 `json:"evidence_recall"`
	Precision      float64 `json:"precision"`
	MRR            float64 `json:"mrr"`
	Questions      int     `json:"questions"`
}

type Acc struct {
	HitRate        float64
	EvidenceRecall float64
	Precision      float64
	MRR            float64
	Questions      int
}

func main() {
	if len(os.Args) != 6 {
		panic("usage: locomo-hippo5-retrieval input.jsonl output.json topk_csv epsilon threshold")
	}
	inputPath := os.Args[1]
	outputPath := os.Args[2]
	topK := parseInts(os.Args[3])
	epsilon := parseFloat32(os.Args[4])
	threshold := parseFloat32(os.Args[5])

	started := time.Now()
	file, err := os.Open(inputPath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	acc := map[int]*Acc{}
	categoryAcc := map[int]map[int]*Acc{}
	for _, k := range topK {
		acc[k] = &Acc{}
	}

	var conversations int
	var recordsTotal int
	var questionsTotal int
	var dimensions int

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024*256)
	for scanner.Scan() {
		var conv Conversation
		if err := json.Unmarshal(scanner.Bytes(), &conv); err != nil {
			panic(err)
		}
		if len(conv.Records) == 0 || len(conv.Questions) == 0 {
			continue
		}
		conversations++
		recordsTotal += len(conv.Records)
		questionsTotal += len(conv.Questions)
		dimensions = len(conv.Records[0].Vector)

		db, err := database.New(dimensions)
		if err != nil {
			panic(err)
		}
		for _, record := range conv.Records {
			if _, err := db.Insert(record.Vector, record.Text, database.Metadata{
				"dia_id": record.DiaID,
			}); err != nil {
				panic(err)
			}
		}

		maxK := maxInt(topK)
		for _, question := range conv.Questions {
			results, err := db.Search(question.Vector, database.SearchOptions{
				Epsilon: epsilon,
				Threshold: threshold,
				TopK: maxK,
			})
			if err != nil {
				panic(err)
			}
			evidence := map[string]bool{}
			for _, diaID := range question.Evidence {
				evidence[diaID] = true
			}
			for _, k := range topK {
				updateAcc(acc[k], results, evidence, k)
				if _, ok := categoryAcc[question.Category]; !ok {
					categoryAcc[question.Category] = map[int]*Acc{}
					for _, categoryK := range topK {
						categoryAcc[question.Category][categoryK] = &Acc{}
					}
				}
				updateAcc(categoryAcc[question.Category][k], results, evidence, k)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}

	topOut := map[string]Metric{}
	for _, k := range topK {
		topOut[strconv.Itoa(k)] = acc[k].metric()
	}

	categoryOut := map[string]map[string]Metric{}
	for category, byK := range categoryAcc {
		categoryOut[strconv.Itoa(category)] = map[string]Metric{}
		for _, k := range topK {
			categoryOut[strconv.Itoa(category)][strconv.Itoa(k)] = byK[k].metric()
		}
	}

	payload := map[string]any{
		"conversations": conversations,
		"questions": questionsTotal,
		"records": recordsTotal,
		"records_per_conversation_mean": float64(recordsTotal) / float64(conversations),
		"dimensions": dimensions,
		"elapsed_seconds": time.Since(started).Seconds(),
		"topk": topOut,
		"by_category": categoryOut,
	}
	encoded, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(outputPath, encoded, 0644); err != nil {
		panic(err)
	}
	fmt.Println(string(encoded))
}

func updateAcc(acc *Acc, results []database.Result, evidence map[string]bool, k int) {
	limit := k
	if limit > len(results) {
		limit = len(results)
	}
	hits := 0
	firstHitRank := 0
	for rank := 0; rank < limit; rank++ {
		diaID, _ := results[rank].Record.Metadata["dia_id"].(string)
		if evidence[diaID] {
			hits++
			if firstHitRank == 0 {
				firstHitRank = rank + 1
			}
		}
	}
	if hits > 0 {
		acc.HitRate++
		acc.MRR += 1.0 / float64(firstHitRank)
	}
	acc.EvidenceRecall += float64(hits) / float64(len(evidence))
	acc.Precision += float64(hits) / float64(k)
	acc.Questions++
}

func (acc *Acc) metric() Metric {
	if acc.Questions == 0 {
		return Metric{}
	}
	n := float64(acc.Questions)
	return Metric{
		HitRate: acc.HitRate / n,
		EvidenceRecall: acc.EvidenceRecall / n,
		Precision: acc.Precision / n,
		MRR: acc.MRR / n,
		Questions: acc.Questions,
	}
}

func parseInts(raw string) []int {
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			panic(err)
		}
		out = append(out, value)
	}
	return out
}

func parseFloat32(raw string) float32 {
	value, err := strconv.ParseFloat(raw, 32)
	if err != nil {
		panic(err)
	}
	return float32(value)
}

func maxInt(values []int) int {
	maxValue := 0
	for _, value := range values {
		if value > maxValue {
			maxValue = value
		}
	}
	return maxValue
}
'''


SESSION_RE = re.compile(r"^session_(\d+)$")


def log(started: float, message: str) -> None:
    print(f"[locomo-hippo5 +{time.perf_counter() - started:.1f}s] {message}", file=sys.stderr, flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
        )


def ensure_locomo(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def load_locomo(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("LoCoMo JSON must be a list of conversations.")
    return payload


def iter_turn_records(sample: dict) -> list[dict]:
    conversation = sample.get("conversation", {})
    records: list[dict] = []
    for session_key in sorted(conversation, key=session_sort_key):
        if not SESSION_RE.match(session_key):
            continue
        turns = conversation.get(session_key)
        if not isinstance(turns, list):
            continue
        session_date = conversation.get(f"{session_key}_date_time", "")
        for turn in turns:
            dia_id = turn.get("dia_id")
            text = turn.get("text", "")
            if not dia_id or not text:
                continue
            speaker = turn.get("speaker", "")
            parts = []
            if session_date:
                parts.append(f"Date: {session_date}")
            if speaker:
                parts.append(f"{speaker}: {text}")
            else:
                parts.append(text)
            caption = turn.get("blip_caption")
            if caption:
                parts.append(f"Image caption: {caption}")
            records.append(
                {
                    "dia_id": str(dia_id),
                    "text": "\n".join(parts),
                }
            )
    return records


def session_sort_key(key: str) -> tuple[int, str]:
    match = SESSION_RE.match(key)
    if match:
        return (int(match.group(1)), key)
    return (10**9, key)


def iter_questions(sample: dict, categories: set[int]) -> list[dict]:
    questions = []
    for index, item in enumerate(sample.get("qa", [])):
        evidence = [str(value) for value in item.get("evidence", []) if value]
        category = int(item.get("category", 0))
        question = item.get("question", "")
        if not evidence or not question or (categories and category not in categories):
            continue
        questions.append(
            {
                "id": str(index),
                "category": category,
                "question": question,
                "evidence": evidence,
            }
        )
    return questions


def collect_benchmark_items(samples: list[dict], categories: set[int], conversation_limit: int | None) -> list[dict]:
    conversations = []
    for sample_index, sample in enumerate(samples):
        records = iter_turn_records(sample)
        questions = iter_questions(sample, categories)
        if not records or not questions:
            continue
        record_ids = {record["dia_id"] for record in records}
        filtered_questions = [
            question
            for question in questions
            if any(evidence_id in record_ids for evidence_id in question["evidence"])
        ]
        if not filtered_questions:
            continue
        conversations.append(
            {
                "id": str(sample.get("sample_id", sample_index)),
                "records": records,
                "questions": filtered_questions,
            }
        )
        if conversation_limit is not None and len(conversations) >= conversation_limit:
            break
    return conversations


def write_go_harness(directory: Path, hippo5_path: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "go.mod").write_text(
        "\n".join(
            [
                "module locomo-hippo5-retrieval",
                "",
                "go 1.18",
                "",
                "require hippo5 v0.0.0",
                "",
                f"replace hippo5 => {hippo5_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (directory / "main.go").write_text(GO_BENCH_SOURCE, encoding="utf-8")
    run(["go", "mod", "tidy"], cwd=directory)
    binary = directory / "locomo-hippo5-retrieval"
    run(["go", "build", "-o", str(binary), "."], cwd=directory)
    return binary


def encode_lookup(
    texts: list[str],
    encoder: StudentEncoder,
    batch_size: int,
    prefix: str,
) -> dict[str, list[float]]:
    prefixed = [f"{prefix}{text}" for text in texts]
    encoded = encoder.encode(prefixed, batch_size=batch_size)
    return {text: encoded[index].tolist() for index, text in enumerate(texts)}


def write_jsonl(
    path: Path,
    conversations: list[dict],
    encoder: StudentEncoder,
    batch_size: int,
    record_prefix: str,
    query_prefix: str,
    started: float,
) -> None:
    record_texts = sorted({record["text"] for conv in conversations for record in conv["records"]})
    question_texts = sorted({question["question"] for conv in conversations for question in conv["questions"]})
    log(started, f"encoding {len(record_texts)} memory records")
    record_vectors = encode_lookup(record_texts, encoder=encoder, batch_size=batch_size, prefix=record_prefix)
    log(started, f"encoding {len(question_texts)} questions")
    question_vectors = encode_lookup(question_texts, encoder=encoder, batch_size=batch_size, prefix=query_prefix)
    with path.open("w", encoding="utf-8") as handle:
        for conv_index, conv in enumerate(conversations, start=1):
            row = {
                "id": conv["id"],
                "records": [
                    {
                        "dia_id": record["dia_id"],
                        "text": record["text"],
                        "vector": record_vectors[record["text"]],
                    }
                    for record in conv["records"]
                ],
                "questions": [
                    {
                        **question,
                        "vector": question_vectors[question["question"]],
                    }
                    for question in conv["questions"]
                ],
            }
            handle.write(json.dumps(row) + "\n")
            log(started, f"wrote conversation {conv_index}/{len(conversations)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo evidence retrieval through the real Hippo-5 Go DB.")
    parser.add_argument("--locomo-json", default="/tmp/locomo10.json")
    parser.add_argument("--locomo-url", default=DEFAULT_LOCOMO_URL)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--hippo5-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conversation-limit", type=int, default=None)
    parser.add_argument("--categories", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--top-k", type=int, nargs="+", default=DEFAULT_TOP_K)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--record-prefix", default="passage: ")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    locomo_path = Path(args.locomo_json)
    if args.download:
        log(started, f"ensuring LoCoMo data at {locomo_path}")
        ensure_locomo(locomo_path, args.locomo_url)
    if not locomo_path.exists():
        raise FileNotFoundError(f"LoCoMo data not found: {locomo_path}. Pass --download to fetch it.")

    samples = load_locomo(locomo_path)
    conversations = collect_benchmark_items(
        samples,
        categories=set(args.categories),
        conversation_limit=args.conversation_limit,
    )
    if not conversations:
        raise ValueError("No LoCoMo conversations with evidence-backed questions were found.")
    question_count = sum(len(conv["questions"]) for conv in conversations)
    record_count = sum(len(conv["records"]) for conv in conversations)
    log(started, f"loaded {len(conversations)} conversations, {record_count} records, {question_count} questions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(started, f"loading student on device={device}")
    encoder = StudentEncoder(args.student_checkpoint, device=device, max_length=args.max_length)

    with tempfile.TemporaryDirectory(dir=args.work_dir) as tmp:
        tmp_path = Path(tmp)
        jsonl_path = tmp_path / "locomo_vectors.jsonl"
        go_dir = tmp_path / "go"
        write_jsonl(
            jsonl_path,
            conversations=conversations,
            encoder=encoder,
            batch_size=args.batch_size,
            record_prefix=args.record_prefix,
            query_prefix=args.query_prefix,
            started=started,
        )
        log(started, "building Go harness")
        binary = write_go_harness(go_dir, hippo5_path=Path(args.hippo5_path).resolve())
        go_output = tmp_path / "go_output.json"
        log(started, "running Hippo-5 retrieval")
        run(
            [
                str(binary),
                str(jsonl_path),
                str(go_output),
                ",".join(str(value) for value in sorted(set(args.top_k))),
                str(args.epsilon),
                str(args.threshold),
            ],
            cwd=go_dir,
        )
        result = json.loads(go_output.read_text(encoding="utf-8"))

    result["config"] = {
        "locomo_json": str(locomo_path),
        "student_checkpoint": args.student_checkpoint,
        "hippo5_path": args.hippo5_path,
        "conversation_limit": args.conversation_limit,
        "categories": args.categories,
        "top_k": sorted(set(args.top_k)),
        "epsilon": args.epsilon,
        "threshold": args.threshold,
        "record_prefix": args.record_prefix,
        "query_prefix": args.query_prefix,
        "device": str(device),
        "python_elapsed_seconds": time.perf_counter() - started,
    }
    encoded = json.dumps(result, indent=2)
    Path(args.output).write_text(encoded, encoding="utf-8")
    log(started, f"wrote output to {args.output}")
    print(encoded)


if __name__ == "__main__":
    main()
