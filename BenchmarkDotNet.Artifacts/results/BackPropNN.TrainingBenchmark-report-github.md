```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean               | Error            | StdDev           | Allocated |
|---------------------- |-------------------:|-----------------:|-----------------:|----------:|
| XOR_SingleIteration   |           416.1 ns |          3.55 ns |          3.32 ns |         - |
| MNIST_SingleIteration | 4,845,556,367.1 ns | 48,971,577.84 ns | 45,808,044.30 ns |     400 B |
