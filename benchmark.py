"""Matmul benchmark: CPU vs MPS (Apple Silicon GPU), with a fallback to CPU-only."""

import json
import platform
import time

import torch


def sync(device: str) -> None:
    # GPU operations are asynchronous: wait until they finish before stopping the timer.
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench(device: str, n: int, repeats: int = 5) -> float:
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    a @ b  # warm-up (first call pays for kernel compilation and memory allocation)
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        sync(device)
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]  # median


def main() -> None:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    print(f"Python  {platform.python_version()}  |  {platform.platform()}")
    print(f"PyTorch {torch.__version__}")
    print(f"MPS available:  {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU threads:    {torch.get_num_threads()}\n")

    sizes = [256, 512, 1000, 2000, 4000]
    results = {d: {} for d in devices}
    print(f"{'n':>6} " + " ".join(f"{d + ' [ms]':>12} {d + ' GFLOPS':>12}" for d in devices))
    for n in sizes:
        row = f"{n:>6} "
        for d in devices:
            t = bench(d, n)
            gflops = 2 * n**3 / t / 1e9  # an n×n matmul is ~2n³ floating-point operations
            results[d][n] = {"ms": t * 1000, "gflops": gflops}
            row += f"{t * 1000:>12.2f} {gflops:>12.1f} "
        print(row)

    with open("results.json", "w") as f:
        json.dump(
            {
                "torch": torch.__version__,
                "platform": platform.platform(),
                "mps": torch.backends.mps.is_available(),
                "results": results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
