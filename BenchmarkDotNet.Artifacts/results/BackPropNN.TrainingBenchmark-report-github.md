```

BenchmarkDotNet v0.13.12, Ubuntu 24.04.2 LTS (Noble Numbat) WSL
AMD Ryzen 9 7950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 9.0.6 (9.0.625.26613), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


```
| Method                | Mean               | Error            | StdDev           | Gen0       | Allocated   |
|---------------------- |-------------------:|-----------------:|-----------------:|-----------:|------------:|
| XOR_SingleIteration   |           468.9 ns |          6.98 ns |          6.52 ns |     0.0324 |       544 B |
| MNIST_SingleIteration | 5,044,458,727.3 ns | 55,241,361.85 ns | 51,672,804.15 ns | 24000.0000 | 406080400 B |
