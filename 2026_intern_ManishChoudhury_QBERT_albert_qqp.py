#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q-BERT Quantization | ALBERT-Base-v2 × QQP
File   : 2026_intern_ManishChoudhury_QBERT_albert_qqp.py
Author : Manish Choudhury (2026 Intern)
"""
import os, gc, copy, time, warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(42)

CHECKPOINT  = "Alireza1044/albert-base-v2-qqp"
WEIGHT_BIT  = 8
MAX_LEN     = 128
RESULTS_CSV = "result_albert_qqp.csv"
LATENCY_WARMUP, LATENCY_RUNS, LATENCY_BATCH = 20, 100, 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print("  Q-BERT | ALBERT-Base-v2 × QQP")
print("="*60)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU : {p.name}  |  VRAM : {p.total_memory/1024**3:.1f} GB")
print(f"  INT : {WEIGHT_BIT}-bit  |  Device : {DEVICE}")
print("="*60, "\n")

class AsymmetricQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, k, x_min, x_max, per_channel=True, _=False):
        if k == 32: return x
        n = float(2**k - 1)
        if per_channel and isinstance(x_min, torch.Tensor) and x_min.numel() > 1:
            x_min = x_min.to(x.device); x_max = x_max.to(x.device)
            s  = ((x_max - x_min) / n).clamp(min=1e-8).unsqueeze(0)
            zp = (-x_min.unsqueeze(0) / s).round().clamp(0, n)
            return ((((x / s).round() + zp).clamp(0, n) - zp) * s).to(x.dtype)
        xmin = x_min.min().item() if isinstance(x_min, torch.Tensor) else float(x_min)
        xmax = x_max.max().item() if isinstance(x_max, torch.Tensor) else float(x_max)
        s = (xmax - xmin) / n
        if s < 1e-8: return x
        zp = max(0, min(int(n), int(round(-xmin / s))))
        return ((((x / s).round() + zp).clamp(0, n) - zp) * s).to(x.dtype)
    @staticmethod
    def backward(ctx, g): return g, None, None, None, None, None

class QuantLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, weight_bit=8, per_channel=True):
        super().__init__(in_f, out_f, bias)
        self.weight_bit = weight_bit; self.per_channel = per_channel
        self._qfn = AsymmetricQuantFunction.apply
    @classmethod
    def from_linear(cls, l, weight_bit=8, per_channel=True):
        inst = cls(l.in_features, l.out_features, l.bias is not None, weight_bit, per_channel)
        inst.weight = nn.Parameter(l.weight.data.clone())
        if l.bias is not None: inst.bias = nn.Parameter(l.bias.data.clone())
        return inst
    def forward(self, x):
        w = self.weight
        if self.per_channel:
            wt = w.data.t().contiguous()
            w_min, w_max = wt.min(dim=1)[0], wt.max(dim=1)[0]
        else:
            w_min, w_max = w.data.min().unsqueeze(0), w.data.max().unsqueeze(0)
        return F.linear(x, self._qfn(w, self.weight_bit, w_min, w_max, self.per_channel, False), self.bias)

def apply_qbert(model, weight_bit=8, per_channel=True):
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, QuantLinear): continue
        parts, parent = name.split("."), model
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], QuantLinear.from_linear(module, weight_bit, per_channel).to(module.weight.device))
        replaced += 1
    print(f"  ✓ Replaced {replaced} Linear → QuantLinear (INT{weight_bit})")
    return model

def fp32_size_mb(model): return sum(p.numel() for p in model.parameters()) * 4 / 1024**2
def int8_size_mb(model, wb=8): return sum(p.numel() * (wb/8 if "weight" in n else 4) for n, p in model.named_parameters()) / 1024**2

def sparsity(model):
    total = zeros = 0
    for n, p in model.named_parameters():
        if "weight" in n: total += p.numel(); zeros += (p.data == 0).sum().item()
    return round(100.0 * zeros / total, 4) if total > 0 else 0.0

def compute_metrics(preds, labels, lat_ms):
    avg = "weighted"
    p  = round(precision_score(labels, preds, average=avg, zero_division=0) * 100, 4)
    r  = round(recall_score(labels,  preds, average=avg, zero_division=0) * 100, 4)
    f1 = round(f1_score(labels,      preds, average=avg, zero_division=0) * 100, 4)
    return p, r, f1, round(1000.0 / lat_ms, 4), round(210.0 * (lat_ms / 1000.0) * 1000.0, 4)

def print_full_results(tag, acc, size_mb, lat_ms, p, r, f1, throughput, spar, energy):
    print(f"\n  [{tag}] Full Metrics:")
    print(f"  {'Accuracy':<28} {acc:.4f} %")
    print(f"  {'Memory Size':<28} {size_mb:.4f} MB")
    print(f"  {'Latency':<28} {lat_ms:.6f} ms/sample")
    print(f"  {'Throughput':<28} {throughput:.4f} samples/sec")
    print(f"  {'% of Sparsity':<28} {spar:.4f} %")
    print(f"  {'Precision':<28} {p:.4f} %")
    print(f"  {'Recall':<28} {r:.4f} %")
    print(f"  {'F1 Score':<28} {f1:.4f} %")
    print(f"  {'Energy Consumption':<28} {energy:.4f} mJ/sample")

def measure_latency(model, tokenizer, t1s, t2s):
    model.eval()
    a = [str(t) for t in t1s[:LATENCY_BATCH]]; b = [str(t) for t in t2s[:LATENCY_BATCH]]
    enc = tokenizer(a, b, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.inference_mode():
        for _ in range(LATENCY_WARMUP): model(**enc)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(LATENCY_RUNS): model(**enc)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000 / (LATENCY_RUNS * len(a))

def evaluate(model, tokenizer, dataset, batch_size=64):
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for i in tqdm(range(0, len(dataset), batch_size), desc="  Evaluating", leave=False):
            b = dataset.select(range(i, min(i + batch_size, len(dataset))))
            t1s = [str(t) for t in b["question1"]]; t2s = [str(t) for t in b["question2"]]
            enc = tokenizer(t1s, t2s, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            all_preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
            all_labels.extend([int(l) for l in b["label"]])
    valid = [(p, l) for p, l in zip(all_preds, all_labels) if l != -1]
    pf, lf = zip(*valid)
    return 100.0 * sum(p == l for p, l in zip(pf, lf)) / len(lf), list(pf), list(lf)

def main():
    print(f"[LOAD]  {CHECKPOINT}")
    tokenizer  = AutoTokenizer.from_pretrained(CHECKPOINT)
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    model_fp32.eval().to(DEVICE)
    print(f"        {sum(p.numel() for p in model_fp32.parameters()):,} parameters\n")
    val_ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
    if len(val_ds) > 3000: val_ds = val_ds.select(range(3000))
    print(f"[DATA]  {len(val_ds)} samples\n")

    print("[FP32]  Evaluating...")
    fp32_acc, fp32_preds, fp32_labels = evaluate(model_fp32, tokenizer, val_ds)
    fp32_size = fp32_size_mb(model_fp32)
    fp32_lat  = measure_latency(model_fp32, tokenizer, val_ds["question1"], val_ds["question2"])
    fp32_spar = sparsity(model_fp32)
    fp32_p, fp32_r, fp32_f1, fp32_thr, fp32_eng = compute_metrics(fp32_preds, fp32_labels, fp32_lat)
    print_full_results("FP32", fp32_acc, fp32_size, fp32_lat, fp32_p, fp32_r, fp32_f1, fp32_thr, fp32_spar, fp32_eng)

    print(f"\n[QBERT] Quantizing...")
    model_q = apply_qbert(copy.deepcopy(model_fp32), WEIGHT_BIT)
    model_q.eval().to(DEVICE)

    print(f"\n[INT8]  Evaluating...")
    q_acc, q_preds, q_labels = evaluate(model_q, tokenizer, val_ds)
    q_size = int8_size_mb(model_q, WEIGHT_BIT)
    q_lat  = measure_latency(model_q, tokenizer, val_ds["question1"], val_ds["question2"])
    q_spar = sparsity(model_q)
    q_p, q_r, q_f1, q_thr, q_eng = compute_metrics(q_preds, q_labels, q_lat)
    print_full_results("INT8", q_acc, q_size, q_lat, q_p, q_r, q_f1, q_thr, q_spar, q_eng)

    acc_drop = round(fp32_acc - q_acc, 2)
    compress = round(fp32_size / q_size, 2)
    speedup  = round(fp32_lat / q_lat, 3)

    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<28} {'FP32':>10}  {'INT8':>10}  {'Delta':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Accuracy (%)':<28} {fp32_acc:>10.4f}  {q_acc:>10.4f}  {acc_drop:>+8.2f}")
    print(f"  {'Memory Size (MB)':<28} {fp32_size:>10.4f}  {q_size:>10.4f}  {compress:>7.2f}x")
    print(f"  {'Latency (ms/sample)':<28} {fp32_lat:>10.6f}  {q_lat:>10.6f}  {speedup:>7.3f}x")
    print(f"  {'Throughput (sps)':<28} {fp32_thr:>10.4f}  {q_thr:>10.4f}")
    print(f"  {'Sparsity (%)':<28} {fp32_spar:>10.4f}  {q_spar:>10.4f}")
    print(f"  {'Precision (%)':<28} {fp32_p:>10.4f}  {q_p:>10.4f}")
    print(f"  {'Recall (%)':<28} {fp32_r:>10.4f}  {q_r:>10.4f}")
    print(f"  {'F1 Score (%)':<28} {fp32_f1:>10.4f}  {q_f1:>10.4f}")
    print(f"  {'Energy (mJ/sample)':<28} {fp32_eng:>10.4f}  {q_eng:>10.4f}")
    print(f"{'='*60}")

    pd.DataFrame([{
        "model":"albert-base-v2","task":"qqp","checkpoint":CHECKPOINT,
        "quantization":f"Q-BERT asymmetric per-channel INT{WEIGHT_BIT}",
        "fp32_accuracy_%":round(fp32_acc,4),"int8_accuracy_%":round(q_acc,4),"accuracy_drop_%":acc_drop,
        "fp32_size_mb":round(fp32_size,4),"int8_size_mb":round(q_size,4),"compression_ratio_x":compress,
        "fp32_latency_ms":round(fp32_lat,6),"int8_latency_ms":round(q_lat,6),"latency_speedup_x":speedup,
        "fp32_throughput_sps":fp32_thr,"int8_throughput_sps":q_thr,
        "fp32_sparsity_%":fp32_spar,"int8_sparsity_%":q_spar,
        "fp32_precision_%":fp32_p,"int8_precision_%":q_p,
        "fp32_recall_%":fp32_r,"int8_recall_%":q_r,
        "fp32_f1_%":fp32_f1,"int8_f1_%":q_f1,
        "fp32_energy_mj":fp32_eng,"int8_energy_mj":q_eng,
        "weight_bit":WEIGHT_BIT,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "server":"edgellm3@10.10.39.108"
    }]).to_csv(RESULTS_CSV, index=False)
    print(f"\n  Results saved → {RESULTS_CSV}")
    del model_fp32, model_q
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

if __name__ == "__main__": main()
