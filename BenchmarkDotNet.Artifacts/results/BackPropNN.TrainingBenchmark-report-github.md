```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean               | Error            | StdDev           | Median             | Allocated |
|---------------------- |-------------------:|-----------------:|-----------------:|-------------------:|----------:|
| XOR_SingleIteration   |           432.9 ns |          7.41 ns |         14.28 ns |           426.5 ns |         - |
| MNIST_SingleIteration | 1,353,687,547.3 ns | 20,999,908.97 ns | 18,615,872.31 ns | 1,347,034,776.5 ns |     400 B |
