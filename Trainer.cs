using System.Diagnostics;

namespace BackPropNN;

class Trainer
{
    private NeuralNetwork _network;
    private double[][] _trainingInputs;
    private double[][] _expectedOutputs;
    private double _learningRate;
    private int _epochs;

    public Trainer(NeuralNetwork network, double[][] trainingInputs, double[][] expectedOutputs, double learningRate, int epochs)
    {
        _network = network;
        _trainingInputs = trainingInputs;
        _expectedOutputs = expectedOutputs;
        _learningRate = learningRate;
        _epochs = epochs;
    }

    public void Train()
    {
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            double totalError = 0;
            var stopWatch = new Stopwatch();
            stopWatch.Start();

            // Train on each input-output pair
            for (int i = 0; i < _trainingInputs.Length; i++)
            {
                var outputs = _network.FeedForward(new Vector(_trainingInputs[i]));
                var errors = new Vector(outputs.Length);
                for (int j = 0; j < outputs.Length; j++)
                {                    
                    errors[j] = _expectedOutputs[i][j] - outputs[j];
                    totalError += Math.Abs(errors[j]);
                }

                _network.BackPropagate(new Vector(_trainingInputs[i]), errors, _learningRate);
            }

            stopWatch.Stop();
            Console.WriteLine($"Epoch={epoch + 1}: Average Abs Error={totalError / _trainingInputs.Length} Time={stopWatch.ElapsedMilliseconds} ms");
        }
    }
}
