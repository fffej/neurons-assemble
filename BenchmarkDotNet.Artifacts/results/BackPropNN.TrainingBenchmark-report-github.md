```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean              | Error           | StdDev          | Gen0         | Gen1         | Allocated       |
|---------------------- |------------------:|----------------:|----------------:|-------------:|-------------:|----------------:|
| XOR_SingleIteration   |          2.462 μs |       0.0297 μs |       0.0263 μs |       0.7935 |            - |           13 KB |
| MNIST_SingleIteration | 12,178,798.764 μs | 232,643.1352 μs | 636,856.8282 μs | 9253000.0000 | 4626000.0000 | 151159688.22 KB |
