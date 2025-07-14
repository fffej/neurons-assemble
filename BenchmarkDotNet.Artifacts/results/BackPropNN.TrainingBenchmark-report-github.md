```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                          | Mean             | Error           | StdDev          | Allocated |
|-------------------------------- |-----------------:|----------------:|----------------:|----------:|
| XOR_SingleIteration             |         431.2 ns |         2.52 ns |         2.23 ns |         - |
| MNIST_SingleIteration           | 996,691,067.0 ns | 8,182,681.00 ns | 7,654,084.89 ns |     736 B |
| XOR_Optimized_SingleIteration   |         296.1 ns |         4.82 ns |         4.27 ns |         - |
| MNIST_Optimized_SingleIteration | 924,531,188.2 ns | 9,391,480.59 ns | 8,784,796.77 ns |     736 B |
