```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean               | Error            | StdDev           | Gen0       | Gen1      | Allocated    |
|---------------------- |-------------------:|-----------------:|-----------------:|-----------:|----------:|-------------:|
| XOR_SingleIteration   |           538.5 ns |          4.27 ns |          3.99 ns |     0.0648 |         - |      1.06 KB |
| MNIST_SingleIteration | 4,927,627,661.5 ns | 94,391,116.14 ns | 78,820,856.97 ns | 48000.0000 | 1000.0000 | 793125.72 KB |
