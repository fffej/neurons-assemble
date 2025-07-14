```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                          | Mean               | Error            | StdDev           | Allocated |
|-------------------------------- |-------------------:|-----------------:|-----------------:|----------:|
| XOR_SingleIteration             |           439.3 ns |          4.44 ns |          3.71 ns |         - |
| MNIST_SingleIteration           | 1,028,196,547.1 ns | 12,663,749.52 ns | 11,845,679.16 ns |     448 B |
| XOR_Optimized_SingleIteration   |           262.0 ns |          1.61 ns |          1.43 ns |         - |
| MNIST_Optimized_SingleIteration | 2,671,388,659.7 ns | 37,529,846.22 ns | 33,269,231.13 ns |      64 B |
