#!/usr/bin/env python3
"""
Benchmark script for MLX STFT/iSTFT implementations.

Tests runtime for all permutations of:
- Backend: FFT vs Conv
- Compilation: Regular vs Compiled (JIT)
- Transform: STFT vs iSTFT
- Spectrum: Onesided vs Dualsided
- Various n_fft sizes and signal lengths
"""

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List

import mlx.core as mx

from mlx_stft import STFT, ISTFT, CompiledSTFT, CompiledISTFT


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    name: str
    n_fft: int
    signal_length: int
    batch_size: int
    backend: str
    compiled: bool
    onesided: bool
    transform: str
    avg_time_ms: float
    std_time_ms: float
    throughput_samples_per_sec: float


def benchmark_fn(
    fn: Callable,
    args: tuple,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> tuple:
    """
    Benchmark a function with warmup and multiple runs.

    Returns:
        tuple: (average_time_ms, std_time_ms)
    """
    # Warmup runs
    for _ in range(warmup_runs):
        result = fn(*args)
        mx.eval(result)

    # Benchmark runs
    times = []
    for _ in range(benchmark_runs):
        mx.synchronize()
        start = time.perf_counter()
        result = fn(*args)
        mx.eval(result)
        mx.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return avg_time, std_time


def run_stft_benchmark(
    n_fft: int,
    signal_length: int,
    batch_size: int,
    use_fft: bool,
    compiled: bool,
    onesided: bool,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> BenchmarkResult:
    """Run STFT benchmark for a specific configuration."""
    hop_length = n_fft // 4

    # Create transform
    if compiled:
        stft = CompiledSTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=onesided,
            use_fft=use_fft,
        )
    else:
        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=onesided,
            use_fft=use_fft,
        )

    # Create input
    x = mx.random.normal((batch_size, signal_length))
    mx.eval(x)

    # Benchmark
    avg_time, std_time = benchmark_fn(
        stft, (x,), warmup_runs, benchmark_runs
    )

    # Calculate throughput
    total_samples = batch_size * signal_length
    throughput = total_samples / (avg_time / 1000)  # samples per second

    backend = "FFT" if use_fft else "Conv"
    name = f"STFT_{backend}_{'Compiled' if compiled else 'Regular'}_{'1sided' if onesided else '2sided'}"

    return BenchmarkResult(
        name=name,
        n_fft=n_fft,
        signal_length=signal_length,
        batch_size=batch_size,
        backend=backend,
        compiled=compiled,
        onesided=onesided,
        transform="STFT",
        avg_time_ms=avg_time,
        std_time_ms=std_time,
        throughput_samples_per_sec=throughput,
    )


def run_istft_benchmark(
    n_fft: int,
    signal_length: int,
    batch_size: int,
    use_fft: bool,
    compiled: bool,
    onesided: bool,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> BenchmarkResult:
    """Run iSTFT benchmark for a specific configuration."""
    hop_length = n_fft // 4

    # Create STFT to generate input spectrum (always use FFT for consistency)
    stft = STFT(
        n_fft=n_fft,
        hop_length=hop_length,
        onesided=onesided,
        use_fft=True,
    )

    # Create iSTFT
    if compiled:
        istft = CompiledISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=onesided,
            use_fft=use_fft,
        )
    else:
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=onesided,
            use_fft=use_fft,
        )

    # Create input spectrum
    x = mx.random.normal((batch_size, signal_length))
    spectrum = stft(x)
    mx.eval(spectrum)

    # Benchmark
    avg_time, std_time = benchmark_fn(
        lambda s: istft(s, length=signal_length),
        (spectrum,),
        warmup_runs,
        benchmark_runs,
    )

    # Calculate throughput
    total_samples = batch_size * signal_length
    throughput = total_samples / (avg_time / 1000)  # samples per second

    backend = "FFT" if use_fft else "Conv"
    name = f"iSTFT_{backend}_{'Compiled' if compiled else 'Regular'}_{'1sided' if onesided else '2sided'}"

    return BenchmarkResult(
        name=name,
        n_fft=n_fft,
        signal_length=signal_length,
        batch_size=batch_size,
        backend=backend,
        compiled=compiled,
        onesided=onesided,
        transform="iSTFT",
        avg_time_ms=avg_time,
        std_time_ms=std_time,
        throughput_samples_per_sec=throughput,
    )


def run_roundtrip_benchmark(
    n_fft: int,
    signal_length: int,
    batch_size: int,
    use_fft: bool,
    compiled: bool,
    onesided: bool,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> BenchmarkResult:
    """Run STFT + iSTFT round-trip benchmark."""
    hop_length = n_fft // 4

    # Note: Conv backend for dualsided has shape mismatch (STFT outputs n_fft+1 bins,
    # iSTFT expects n_fft bins). For roundtrip, use same backend for both.
    # When use_fft=False and onesided=False, we skip as it's incompatible.
    if not use_fft and not onesided:
        raise ValueError("Conv backend roundtrip not supported for dualsided spectrum (shape mismatch)")

    # Create transforms
    if compiled:
        stft = CompiledSTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=onesided,
            use_fft=use_fft,
        )
        istft = CompiledISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=onesided,
            use_fft=use_fft,
        )
    else:
        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=onesided,
            use_fft=use_fft,
        )
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=onesided,
            use_fft=use_fft,
        )

    # Create input
    x = mx.random.normal((batch_size, signal_length))
    mx.eval(x)

    def roundtrip(signal):
        spectrum = stft(signal)
        return istft(spectrum, length=signal_length)

    # Benchmark
    avg_time, std_time = benchmark_fn(
        roundtrip, (x,), warmup_runs, benchmark_runs
    )

    # Calculate throughput
    total_samples = batch_size * signal_length
    throughput = total_samples / (avg_time / 1000)  # samples per second

    backend = "FFT" if use_fft else "Conv"
    name = f"Roundtrip_{backend}_{'Compiled' if compiled else 'Regular'}_{'1sided' if onesided else '2sided'}"

    return BenchmarkResult(
        name=name,
        n_fft=n_fft,
        signal_length=signal_length,
        batch_size=batch_size,
        backend=backend,
        compiled=compiled,
        onesided=onesided,
        transform="Roundtrip",
        avg_time_ms=avg_time,
        std_time_ms=std_time,
        throughput_samples_per_sec=throughput,
    )


def print_results_table(results: List[BenchmarkResult], title: str = ""):
    """Print benchmark results in a formatted table."""
    if title:
        print(f"\n{'=' * 100}")
        print(f" {title}")
        print(f"{'=' * 100}")

    # Header
    print(f"{'Transform':<12} {'Backend':<6} {'Compiled':<9} {'Onesided':<9} "
          f"{'Time (ms)':<14} {'Throughput':<18}")
    print("-" * 100)

    for r in results:
        compiled_str = "Yes" if r.compiled else "No"
        onesided_str = "Yes" if r.onesided else "No"
        time_str = f"{r.avg_time_ms:.3f} +/- {r.std_time_ms:.3f}"
        throughput_str = f"{r.throughput_samples_per_sec / 1e6:.2f} M samples/s"

        print(f"{r.transform:<12} {r.backend:<6} {compiled_str:<9} {onesided_str:<9} "
              f"{time_str:<14} {throughput_str:<18}")


def run_all_benchmarks(
    n_fft_sizes: List[int],
    signal_lengths: List[int],
    batch_sizes: List[int],
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    include_roundtrip: bool = True,
) -> List[BenchmarkResult]:
    """Run all benchmark permutations."""
    all_results = []

    for batch_size in batch_sizes:
        for signal_length in signal_lengths:
            for n_fft in n_fft_sizes:
                print(f"\nBenchmarking: batch={batch_size}, signal_length={signal_length}, n_fft={n_fft}")

                results = []

                # All permutations of backend, compiled, onesided
                for use_fft in [True, False]:
                    for compiled in [False, True]:
                        for onesided in [True, False]:
                            # STFT benchmark
                            try:
                                result = run_stft_benchmark(
                                    n_fft=n_fft,
                                    signal_length=signal_length,
                                    batch_size=batch_size,
                                    use_fft=use_fft,
                                    compiled=compiled,
                                    onesided=onesided,
                                    warmup_runs=warmup_runs,
                                    benchmark_runs=benchmark_runs,
                                )
                                results.append(result)
                            except Exception as e:
                                print(f"  STFT failed: {e}")

                            # iSTFT benchmark
                            try:
                                result = run_istft_benchmark(
                                    n_fft=n_fft,
                                    signal_length=signal_length,
                                    batch_size=batch_size,
                                    use_fft=use_fft,
                                    compiled=compiled,
                                    onesided=onesided,
                                    warmup_runs=warmup_runs,
                                    benchmark_runs=benchmark_runs,
                                )
                                results.append(result)
                            except Exception as e:
                                print(f"  iSTFT failed: {e}")

                            # Roundtrip benchmark
                            if include_roundtrip:
                                try:
                                    result = run_roundtrip_benchmark(
                                        n_fft=n_fft,
                                        signal_length=signal_length,
                                        batch_size=batch_size,
                                        use_fft=use_fft,
                                        compiled=compiled,
                                        onesided=onesided,
                                        warmup_runs=warmup_runs,
                                        benchmark_runs=benchmark_runs,
                                    )
                                    results.append(result)
                                except Exception as e:
                                    print(f"  Roundtrip failed: {e}")

                # Print results for this configuration
                title = f"batch_size={batch_size}, signal_length={signal_length}, n_fft={n_fft}"
                print_results_table(results, title)
                all_results.extend(results)

    return all_results


def print_summary(results: List[BenchmarkResult]):
    """Print summary comparing backends and compilation modes."""
    print("\n" + "=" * 100)
    print(" SUMMARY")
    print("=" * 100)

    # Group by transform type
    transforms = ["STFT", "iSTFT", "Roundtrip"]

    for transform in transforms:
        transform_results = [r for r in results if r.transform == transform]
        if not transform_results:
            continue

        print(f"\n{transform} Summary:")
        print("-" * 50)

        # Compare FFT vs Conv (averaged across all configs)
        fft_results = [r for r in transform_results if r.backend == "FFT"]
        conv_results = [r for r in transform_results if r.backend == "Conv"]

        if fft_results and conv_results:
            fft_avg = sum(r.avg_time_ms for r in fft_results) / len(fft_results)
            conv_avg = sum(r.avg_time_ms for r in conv_results) / len(conv_results)
            speedup = conv_avg / fft_avg if fft_avg > 0 else 0
            print(f"  FFT vs Conv: FFT is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

        # Compare Compiled vs Regular
        compiled_results = [r for r in transform_results if r.compiled]
        regular_results = [r for r in transform_results if not r.compiled]

        if compiled_results and regular_results:
            compiled_avg = sum(r.avg_time_ms for r in compiled_results) / len(compiled_results)
            regular_avg = sum(r.avg_time_ms for r in regular_results) / len(regular_results)
            speedup = regular_avg / compiled_avg if compiled_avg > 0 else 0
            print(f"  Compiled vs Regular: Compiled is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

        # Compare Onesided vs Dualsided
        onesided_results = [r for r in transform_results if r.onesided]
        dualsided_results = [r for r in transform_results if not r.onesided]

        if onesided_results and dualsided_results:
            onesided_avg = sum(r.avg_time_ms for r in onesided_results) / len(onesided_results)
            dualsided_avg = sum(r.avg_time_ms for r in dualsided_results) / len(dualsided_results)
            speedup = dualsided_avg / onesided_avg if onesided_avg > 0 else 0
            print(f"  Onesided vs Dualsided: Onesided is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX STFT/iSTFT implementations")
    parser.add_argument(
        "--n-fft",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="FFT sizes to benchmark (default: 512 1024 2048 4096)",
    )
    parser.add_argument(
        "--signal-length",
        type=int,
        nargs="+",
        default=[16000, 48000],
        help="Signal lengths to benchmark (default: 16000 48000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Batch sizes to benchmark (default: 1 8)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--no-roundtrip",
        action="store_true",
        help="Skip roundtrip benchmarks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark with reduced configurations",
    )

    args = parser.parse_args()

    if args.quick:
        n_fft_sizes = [1024]
        signal_lengths = [16000]
        batch_sizes = [1]
    else:
        n_fft_sizes = args.n_fft
        signal_lengths = args.signal_length
        batch_sizes = args.batch_size

    print("MLX STFT/iSTFT Benchmark")
    print("=" * 100)
    print(f"n_fft sizes: {n_fft_sizes}")
    print(f"Signal lengths: {signal_lengths}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Benchmark runs: {args.benchmark_runs}")
    print(f"Include roundtrip: {not args.no_roundtrip}")

    results = run_all_benchmarks(
        n_fft_sizes=n_fft_sizes,
        signal_lengths=signal_lengths,
        batch_sizes=batch_sizes,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        include_roundtrip=not args.no_roundtrip,
    )

    print_summary(results)

    print("\n" + "=" * 100)
    print(" Benchmark complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
