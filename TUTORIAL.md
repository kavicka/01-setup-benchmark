# Tutorial · 01 · Setup & matmul benchmark

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial rebuilds the project from an empty folder, one step at a time. For every step it says **what** we do, **why** we do it that way, and **how** it works underneath. The README is the report (numbers, plots); this file is the "how did I get there".

**What you end up with:** a reproducible Python environment and a script that measures how fast PyTorch multiplies matrices on the CPU and on the Mac GPU (MPS).

**You need:** a terminal, [`uv`](https://docs.astral.sh/uv/) installed, ~2 GB of disk for PyTorch.

---

## Step 1 · Create the project with `uv`

```bash
uv init setup-benchmark --python 3.12
cd setup-benchmark
uv add torch numpy matplotlib
```

**What happens:**
- `uv init` creates `pyproject.toml` (project name + list of dependencies) and `.python-version` (pins 3.12).
- `uv add` downloads the packages into a private `.venv/` folder and writes **`uv.lock`**.

**Why `uv` and a lockfile:** `pyproject.toml` says "I need torch ≥ 2.15". That is a *range*. `uv.lock` records the *exact* version of every package, including the ~30 packages torch pulls in. On another computer, `uv sync` recreates the identical environment. Without a lock, two machines can install different versions and get different results or errors, and you cannot tell which.

**Why a separate environment per project:** each step in this repo has different dependencies (later ones need scikit-learn, MLX, …). Separate `.venv/` folders mean upgrading a package for step 12 can never break step 01.

Check it works:
```bash
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```
`uv run` runs the command inside `.venv` without you activating anything.

---

## Step 2 · Find out which devices exist

```python
devices = ["cpu"]
if torch.backends.mps.is_available():   # Apple Silicon GPU
    devices.append("mps")
if torch.cuda.is_available():           # NVIDIA GPU
    devices.append("cuda")
```

**How PyTorch thinks about hardware:** every tensor lives on a *device*. `torch.randn(3, 3, device="mps")` creates the numbers directly in GPU memory. An operation runs on the device where its inputs live. So the same code runs on a laptop, a Mac or Colab: only the `device=` string changes.

- **MPS** (Metal Performance Shaders) is Apple's GPU backend.
- **CUDA** is NVIDIA's. On a Mac it is never available, but keeping the check means the script also works on Colab/Kaggle.

**Why detect instead of hard-coding:** a script that says `device="mps"` crashes on any non-Mac. Detecting and falling back to CPU is the pattern every later project uses.

---

## Step 3 · Time one multiplication correctly

This is the heart of the project. The naive version is wrong:

```python
start = time.perf_counter()
a @ b
print(time.perf_counter() - start)   # ❌ wrong on GPU
```

The real function in [`benchmark.py`](benchmark.py) fixes three problems:

```python
def bench(device, n, repeats=5):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    a @ b              # (1) warm-up, not timed
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        sync(device)   # (2) wait for the GPU to finish
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]   # (3) median
```

**(1) Warm-up.** The first call is special: PyTorch allocates memory and the GPU driver prepares (compiles/loads) the kernel. That one-time cost has nothing to do with how fast multiplication is, so we run it once and throw it away.

**(2) Synchronization.** On a GPU, `a @ b` does **not** compute anything before returning. It puts the work in a queue and returns immediately, so Python can keep going while the GPU works in parallel. If you stop the clock right away you measure "how long it took to queue the work", which is microseconds no matter how big the matrix is. `torch.mps.synchronize()` blocks until the queue is empty. On CPU the work is synchronous, so `sync` does nothing:

```python
def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
```

**(3) Median of 5.** Any single measurement can be disturbed by other programs, the OS, or the CPU changing its clock speed. The median ignores one or two outliers; a mean would be pulled by them.

`time.perf_counter()` is used because it is the highest-resolution clock Python has and never jumps (unlike `time.time()`, which follows the wall clock).

---

## Step 4 · Turn time into GFLOPS

Time alone cannot compare a 256×256 matrix with a 4000×4000 one. So we count the work:

- The result has n × n cells.
- Each cell is a dot product of a row and a column: n multiplications + n additions = 2n operations.
- Total: n² · 2n = **2n³ floating-point operations (FLOPs)**.

```python
gflops = 2 * n**3 / t / 1e9
```

FLOPs ÷ seconds = FLOPs per second; ÷ 10⁹ = **GFLOPS** (billions per second). It measures *throughput*: how much useful work the device finishes per second, independent of matrix size.

**Why n³ matters:** double the side length and the work grows 2³ = 8×. This cubic growth is why matrix size (= model width) is so expensive, and later why attention over long contexts is expensive.

---

## Step 5 · Loop over sizes and save raw results

```python
sizes = [256, 512, 1000, 2000, 4000]
for n in sizes:
    for d in devices:
        t = bench(d, n)
        results[d][n] = {"ms": t * 1000, "gflops": 2 * n**3 / t / 1e9}
json.dump({"torch": ..., "platform": ..., "mps": ..., "results": results}, f)
```

The sizes span small to large on purpose, because the GPU's behaviour changes with size (Step 7).

**Why write `results.json` instead of only printing:** measurement and presentation are separate jobs. The benchmark is slow and depends on the machine; plotting is fast and you may restyle it ten times. Saving raw numbers (plus the torch version and platform, so you know where they came from) lets you redraw without re-measuring, and lets the README quote exact numbers. Every later project follows this `script → results.json → plot.py → assets/` pattern.

---

## Step 6 · Plot the results

[`plot.py`](plot.py) reads `results.json` and draws two panels:

- **Left, log-log time vs n,** plus a dashed `n³` reference line anchored at the largest size. On log-log axes, a power law `t ∝ n³` becomes a straight line with slope 3, so you can see at a glance whether the measurement follows theory.
- **Right, GFLOPS vs n.** This shows *efficiency*: how close each device gets to its peak.

```bash
uv run python benchmark.py   # measure → results.json
uv run python plot.py        # draw → assets/matmul_benchmark.png
```

---

## Step 7 · Read the result (what it teaches)

From the Mac run (`results.json`):

| n | CPU GFLOPS | MPS GFLOPS |
|---:|---:|---:|
| 256 | 1,286 | 78 |
| 4000 | 2,929 | 5,233 |

**Why the GPU loses on small matrices:** every GPU call has a fixed overhead (sending the command, launching the kernel, waiting for sync), roughly 0.4 ms here. A 256×256 multiply is so little work that the overhead dominates. With 4000×4000 the work is 4000× bigger, the overhead becomes negligible, and the GPU's thousands of parallel units win.

**Why the CPU is surprisingly fast:** PyTorch calls a highly optimised linear-algebra library (BLAS) that uses the CPU's vector/matrix units on all cores. For small and medium sizes that beats the GPU's launch overhead.

**The lesson for later projects:** GPUs pay off when you give them *big batches of work at once*. That is why training uses mini-batches, why the Racing Lab computes 200 car brains in one matmul (ch01), and why small models on MPS sometimes show no speed-up (project 12's KV-cache measurement).

---

## Try it yourself

1. Delete `sync(device)` from the loop and rerun. MPS will report absurdly high GFLOPS. That is the bug the sync prevents.
2. Add `dtype=torch.float16` to `torch.randn` for MPS. Half precision moves half the bytes; does throughput go up?
3. Add `n = 8000`. Does the CPU keep up, or does its GFLOPS drop (memory/cache limits)?

## Recap

| Step | Idea |
|---|---|
| 1 | `uv` + `uv.lock` = same environment everywhere |
| 2 | Tensors live on a device; detect, don't hard-code |
| 3 | Warm-up, synchronize, median: the three rules of timing GPU code |
| 4 | Matmul = 2n³ FLOPs; GFLOPS makes sizes comparable |
| 5–6 | Save raw numbers, plot separately |
| 7 | GPUs need big batched work to beat their overhead |
