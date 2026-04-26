from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_hippo5_softbox_retrieval import (
    DEFAULT_PRESET,
    DEFAULT_SCORE_THRESHOLDS,
    DEFAULT_TOP_K,
    TeacherEncoder,
    StudentEncoder,
    build_retrieval_cases,
    collect_text_pool,
    encode_texts,
    load_cases,
    prepare_case_region,
)


def log_step(started: float, message: str) -> None:
    print(f"[hippo5-go-benchmark +{time.perf_counter() - started:.1f}s] {message}", file=sys.stderr, flush=True)


GO_BENCH_SOURCE = r'''
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"hippo5/database"
)

type Case struct {
	Query       string   `json:"query"`
	QueryVector []float32 `json:"query_vector"`
	Minus       []float32 `json:"minus"`
	Plus        []float32 `json:"plus"`
	Records     []Record `json:"records"`
}

type Record struct {
	Label  string    `json:"label"`
	Text   string    `json:"text"`
	Vector []float32 `json:"vector"`
}

type Metric struct {
	Accuracy          float64 `json:"accuracy,omitempty"`
	HitRate           float64 `json:"hit_rate,omitempty"`
	Precision         float64 `json:"precision"`
	Recall            float64 `json:"recall"`
	AveragePrecision  float64 `json:"average_precision,omitempty"`
	Coverage          float64 `json:"coverage,omitempty"`
	FalsePositiveRate float64 `json:"false_positive_rate,omitempty"`
	F1                float64 `json:"f1,omitempty"`
	AvgResults        float64 `json:"avg_results,omitempty"`
}

type TopKAcc struct {
	Accuracy float64
	HitRate  float64
	Precision float64
	Recall float64
	AveragePrecision float64
	N float64
}

type ThresholdAcc struct {
	Coverage float64
	Precision float64
	Recall float64
	FalsePositiveRate float64
	F1 float64
	AvgResults float64
	N float64
}

func main() {
	if len(os.Args) != 5 {
		panic("usage: hippo5-go-softbox input.jsonl output.json topk_csv thresholds_csv")
	}
	inputPath := os.Args[1]
	outputPath := os.Args[2]
	topK := parseInts(os.Args[3])
	thresholds := parseFloat32s(os.Args[4])

	started := time.Now()
	file, err := os.Open(inputPath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	topAcc := map[int]*TopKAcc{}
	for _, k := range topK {
		topAcc[k] = &TopKAcc{}
	}
	thresholdAcc := map[float32]*ThresholdAcc{}
	for _, threshold := range thresholds {
		thresholdAcc[threshold] = &ThresholdAcc{}
	}

	var cases int
	var dimensions int
	var recordsTotal int
	var positiveScoreSum float64
	var negativeScoreSum float64
	var positiveScoreCount float64
	var negativeScoreCount float64

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024*256)
	for scanner.Scan() {
		var c Case
		if err := json.Unmarshal(scanner.Bytes(), &c); err != nil {
			panic(err)
		}
		cases++
		dimensions = len(c.QueryVector)
		recordsTotal += len(c.Records)

		db, err := database.New(dimensions)
		if err != nil {
			panic(err)
		}
		positiveCount := 0
		negativeCount := 0
		for _, record := range c.Records {
			if record.Label == "positive" {
				positiveCount++
			} else {
				negativeCount++
			}
			if _, err := db.Insert(record.Vector, record.Text, database.Metadata{"label": record.Label}); err != nil {
				panic(err)
			}
		}

		full, err := db.SearchSoftBox(c.QueryVector, database.SoftBoxSearchOptions{
			Minus: c.Minus,
			Plus: c.Plus,
			TopK: len(c.Records),
			TieBreakByAnchorDistance: true,
		})
		if err != nil {
			panic(err)
		}
		labels := make([]int, len(full))
		for i, result := range full {
			if result.Record.Metadata["label"] == "positive" {
				labels[i] = 1
				positiveScoreSum += float64(result.Score)
				positiveScoreCount++
			} else {
				negativeScoreSum += float64(result.Score)
				negativeScoreCount++
			}
		}

		for _, k := range topK {
			hits := 0
			apHits := 0
			apSum := 0.0
			limit := k
			if limit > len(labels) {
				limit = len(labels)
			}
			for rank := 0; rank < limit; rank++ {
				if labels[rank] == 1 {
					hits++
					apHits++
					apSum += float64(apHits) / float64(rank+1)
				}
			}
			acc := topAcc[k]
			if len(labels) > 0 && labels[0] == 1 {
				acc.Accuracy++
			}
			if hits > 0 {
				acc.HitRate++
			}
			acc.Precision += float64(hits) / float64(k)
			acc.Recall += float64(hits) / float64(positiveCount)
			acc.AveragePrecision += apSum / float64(positiveCount)
			acc.N++
		}

		for _, threshold := range thresholds {
			results, err := db.SearchSoftBox(c.QueryVector, database.SoftBoxSearchOptions{
				Minus: c.Minus,
				Plus: c.Plus,
				ScoreThreshold: &threshold,
				TopK: len(c.Records),
				TieBreakByAnchorDistance: true,
			})
			if err != nil {
				panic(err)
			}
			hits := 0
			for _, result := range results {
				if result.Record.Metadata["label"] == "positive" {
					hits++
				}
			}
			selected := len(results)
			falsePositive := selected - hits
			precision := 0.0
			if selected > 0 {
				precision = float64(hits) / float64(selected)
			}
			recall := float64(hits) / float64(positiveCount)
			f1 := 0.0
			if precision + recall > 0 {
				f1 = 2 * precision * recall / (precision + recall)
			}
			acc := thresholdAcc[threshold]
			if selected > 0 {
				acc.Coverage++
			}
			acc.Precision += precision
			acc.Recall += recall
			acc.FalsePositiveRate += float64(falsePositive) / math.Max(1, float64(negativeCount))
			acc.F1 += f1
			acc.AvgResults += float64(selected)
			acc.N++
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}

	topOut := map[string]Metric{}
	for _, k := range topK {
		acc := topAcc[k]
		topOut[strconv.Itoa(k)] = Metric{
			Accuracy: acc.Accuracy / acc.N,
			HitRate: acc.HitRate / acc.N,
			Precision: acc.Precision / acc.N,
			Recall: acc.Recall / acc.N,
			AveragePrecision: acc.AveragePrecision / acc.N,
		}
	}

	thresholdOut := map[string]Metric{}
	for _, threshold := range thresholds {
		acc := thresholdAcc[threshold]
		thresholdOut[formatFloat(float64(threshold))] = Metric{
			Coverage: acc.Coverage / acc.N,
			Precision: acc.Precision / acc.N,
			Recall: acc.Recall / acc.N,
			FalsePositiveRate: acc.FalsePositiveRate / acc.N,
			F1: acc.F1 / acc.N,
			AvgResults: acc.AvgResults / acc.N,
		}
	}

	positiveMean := 0.0
	negativeMean := 0.0
	if positiveScoreCount > 0 {
		positiveMean = positiveScoreSum / positiveScoreCount
	}
	if negativeScoreCount > 0 {
		negativeMean = negativeScoreSum / negativeScoreCount
	}

	payload := map[string]any{
		"cases": cases,
		"dimensions": dimensions,
		"records_per_case_mean": float64(recordsTotal) / float64(cases),
		"elapsed_seconds": time.Since(started).Seconds(),
		"topk": topOut,
		"thresholds": thresholdOut,
		"score_stats": map[string]float64{
			"positive_mean": positiveMean,
			"negative_mean": negativeMean,
			"gap": negativeMean - positiveMean,
		},
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

func parseFloat32s(raw string) []float32 {
	parts := strings.Split(raw, ",")
	out := make([]float32, 0, len(parts))
	for _, part := range parts {
		if part == "" {
			continue
		}
		value, err := strconv.ParseFloat(part, 32)
		if err != nil {
			panic(err)
		}
		out = append(out, float32(value))
	}
	return out
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'g', -1, 64)
}
'''


def run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
        )


def write_go_harness(directory: Path, hippo5_path: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "go.mod").write_text(
        "\n".join(
            [
                "module hippo5-go-softbox-benchmark",
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
    binary = directory / "hippo5-go-softbox-benchmark"
    run(["go", "build", "-o", str(binary), "."], cwd=directory)
    return binary


def write_jsonl_cases(
    output_path: Path,
    retrieval_cases,
    text_to_index: dict[str, int],
    teacher_embeds: torch.Tensor,
    student_embeds: torch.Tensor,
    radius_scale: float,
    args: argparse.Namespace,
) -> None:
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as handle:
        total_cases = len(retrieval_cases)
        for case_index, case in enumerate(retrieval_cases, start=1):
            prepared = prepare_case_region(
                case,
                text_to_index=text_to_index,
                teacher_embeds=teacher_embeds,
                student_embeds=student_embeds,
                radius_scale=radius_scale,
                args=args,
            )
            records = []
            for text, label in zip(prepared["records"], prepared["labels"], strict=True):
                records.append(
                    {
                        "label": "positive" if label else "negative",
                        "text": text,
                        "vector": student_embeds[text_to_index[text]].tolist(),
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "query": case.query,
                        "query_vector": prepared["query_student"].tolist(),
                        "minus": prepared["minus"].tolist(),
                        "plus": prepared["plus"].tolist(),
                        "records": records,
                    }
                )
                + "\n"
            )
            if case_index == 1 or case_index == total_cases or case_index % 10 == 0:
                print(
                    f"[hippo5-go-benchmark +{time.perf_counter() - started:.1f}s] "
                    f"wrote JSONL case {case_index}/{total_cases}",
                    file=sys.stderr,
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real Hippo-5 Go soft-box retrieval benchmark.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--hippo5-path", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-limit", type=int, default=100)
    parser.add_argument("--positives-per-case", type=int, default=20)
    parser.add_argument("--distractors-per-case", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, nargs="+", default=[*DEFAULT_TOP_K, 20, 50, 100])
    parser.add_argument("--score-thresholds", type=float, nargs="+", default=DEFAULT_SCORE_THRESHOLDS)
    parser.add_argument("--terms-per-side", type=int, default=DEFAULT_PRESET["terms_per_side"])
    parser.add_argument("--min-radius", type=float, default=DEFAULT_PRESET["min_radius"])
    parser.add_argument("--radius-scale", type=float, default=DEFAULT_PRESET["radius_scale"])
    parser.add_argument("--negative-weight", type=float, default=DEFAULT_PRESET["negative_weight"])
    parser.add_argument("--size-weight", type=float, default=DEFAULT_PRESET["size_weight"])
    parser.add_argument("--teacher-weight", type=float, default=DEFAULT_PRESET["teacher_weight"])
    parser.add_argument("--student-weight", type=float, default=DEFAULT_PRESET["student_weight"])
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_step(started, f"using device={device}")
    all_source_cases = load_cases(args.cases, seed=args.seed)
    source_cases = all_source_cases[: args.case_limit] if args.case_limit is not None else all_source_cases
    text_pool = collect_text_pool(all_source_cases)
    log_step(
        started,
        f"loaded {len(source_cases)} benchmark cases from {len(all_source_cases)} source cases; "
        f"text_pool={len(text_pool)}",
    )

    log_step(started, f"loading teacher={args.teacher_model}")
    teacher = TeacherEncoder(args.teacher_model, device=device, max_length=args.max_length)
    log_step(started, f"loading student checkpoint={args.student_checkpoint}")
    student = StudentEncoder(args.student_checkpoint, device=device, max_length=args.max_length)

    pool_teacher_embeds = None
    pool_text_to_index = None
    if args.positives_per_case > 2:
        log_step(started, f"encoding teacher pool for positive expansion: {len(text_pool)} texts")
        pool_text_to_index = {text: index for index, text in enumerate(text_pool)}
        pool_teacher_embeds = teacher.encode(text_pool, batch_size=args.batch_size)
        log_step(started, "teacher pool encoded")

    log_step(started, "building retrieval cases")
    retrieval_cases = build_retrieval_cases(
        cases=source_cases,
        text_pool=text_pool,
        distractors_per_case=args.distractors_per_case,
        positives_per_case=args.positives_per_case,
        teacher_embeds=pool_teacher_embeds,
        text_to_index=pool_text_to_index,
        seed=args.seed + 1,
    )
    log_step(started, f"built {len(retrieval_cases)} retrieval cases")
    all_texts = collect_text_pool(
        {
            "query": case.query,
            "positives": case.positives,
            "negatives": [*case.negatives, *case.distractors],
        }
        for case in retrieval_cases
    )
    log_step(started, f"encoding benchmark texts: {len(all_texts)} unique texts")
    text_to_index, teacher_embeds, student_embeds = encode_texts(
        all_texts,
        teacher=teacher,
        student=student,
        batch_size=args.batch_size,
    )
    log_step(started, "benchmark texts encoded")

    with tempfile.TemporaryDirectory(dir=args.work_dir) as tmp:
        tmp_path = Path(tmp)
        jsonl_path = tmp_path / "cases.jsonl"
        go_dir = tmp_path / "go"
        log_step(started, f"writing temporary JSONL cases to {jsonl_path}")
        write_jsonl_cases(
            jsonl_path,
            retrieval_cases=retrieval_cases,
            text_to_index=text_to_index,
            teacher_embeds=teacher_embeds,
            student_embeds=student_embeds,
            radius_scale=args.radius_scale,
            args=args,
        )
        log_step(started, "building temporary Go benchmark harness")
        binary = write_go_harness(go_dir, hippo5_path=Path(args.hippo5_path).resolve())
        go_output = tmp_path / "go_output.json"
        log_step(started, "running actual Hippo-5 SearchSoftBox benchmark")
        run(
            [
                str(binary),
                str(jsonl_path),
                str(go_output),
                ",".join(str(value) for value in sorted(set(args.top_k))),
                ",".join(str(value) for value in args.score_thresholds),
            ],
            cwd=go_dir,
        )
        log_step(started, "Hippo-5 benchmark completed")
        result = json.loads(go_output.read_text(encoding="utf-8"))

    result["config"] = {
        "cases": args.cases,
        "student_checkpoint": args.student_checkpoint,
        "hippo5_path": args.hippo5_path,
        "teacher_model": args.teacher_model,
        "case_limit": args.case_limit,
        "positives_per_case": args.positives_per_case,
        "distractors_per_case": args.distractors_per_case,
        "radius_scale": args.radius_scale,
        "top_k": sorted(set(args.top_k)),
        "score_thresholds": args.score_thresholds,
        "device": str(device),
        "python_elapsed_seconds": time.perf_counter() - started,
    }
    encoded = json.dumps(result, indent=2)
    Path(args.output).write_text(encoded, encoding="utf-8")
    log_step(started, f"wrote output to {args.output}")
    print(encoded)


if __name__ == "__main__":
    main()
