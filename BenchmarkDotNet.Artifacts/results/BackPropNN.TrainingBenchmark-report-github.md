```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean               | Error            | StdDev            | Median             | Allocated |
|---------------------- |-------------------:|-----------------:|------------------:|-------------------:|----------:|
| XOR_SingleIteration   |           331.7 ns |          6.58 ns |          11.17 ns |           327.0 ns |         - |
| MNIST_SingleIteration | 4,070,880,951.5 ns | 80,602,026.88 ns | 151,390,144.03 ns | 3,982,509,034.5 ns |     400 B |
