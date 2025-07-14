```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean             | Error           | StdDev          | Allocated |
|---------------------- |-----------------:|----------------:|----------------:|----------:|
| XOR_SingleIteration   |         435.3 ns |         3.78 ns |         3.54 ns |         - |
| MNIST_SingleIteration | 953,730,559.7 ns | 9,730,796.78 ns | 8,125,656.02 ns |     448 B |
