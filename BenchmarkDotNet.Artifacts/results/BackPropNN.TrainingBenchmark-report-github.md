```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean               | Error             | StdDev            | Gen0         | Gen1      | Allocated      |
|---------------------- |-------------------:|------------------:|------------------:|-------------:|----------:|---------------:|
| XOR_SingleIteration   |           660.1 ns |           7.02 ns |           6.57 ns |       0.1259 |         - |        2.06 KB |
| MNIST_SingleIteration | 6,279,707,147.6 ns | 120,250,224.85 ns | 112,482,134.94 ns | 1176000.0000 | 4000.0000 | 19208437.94 KB |
