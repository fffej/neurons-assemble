```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean               | Error            | StdDev            | Allocated |
|---------------------- |-------------------:|-----------------:|------------------:|----------:|
| XOR_SingleIteration   |           333.0 ns |          5.23 ns |           4.37 ns |         - |
| MNIST_SingleIteration | 4,110,351,987.7 ns | 81,670,414.09 ns | 187,651,357.96 ns |     400 B |
