using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace BackPropNN;

[MemoryDiagnoser]
public class TrainingBenchmark
{
    private Trainer? _xorTrainer;
    private Trainer? _mnistTrainer;

    [GlobalSetup]
    public void Setup()
    {
        // Setup XOR trainer
        var xorNetwork = new NeuralNetworkImpl(new NeuralNetworkFactoryImpl(), 2, 3, 1);
        double[][] xorInputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        double[][] xorExpectedOutputs = [ [0], [1], [1],  [0] ];
        _xorTrainer = new Trainer(xorNetwork, xorInputs, xorExpectedOutputs, 0.1, 1);

        // Setup MNIST trainer
        var (trainingInputs, trainingLabels) = MNistLoader.Load();
        var mnistNetwork = new NeuralNetworkImpl(new NeuralNetworkFactoryImpl(), 784, 50, 10);
        _mnistTrainer = new Trainer(mnistNetwork, trainingInputs, trainingLabels, 0.1, 1);
    }

    [Benchmark]
    public void XOR_SingleIteration()
    {
        _xorTrainer!.TrainSingleIteration();
    }

    [Benchmark]
    public void MNIST_SingleIteration()
    {
        _mnistTrainer!.TrainSingleIteration();
    }
} 