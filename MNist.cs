using MNIST.IO;

public class MNistLoader
{
    public static (double[][], double[][]) Load()
    {
        var data = FileReaderMNIST.LoadImagesAndLables("data/train-labels-idx1-ubyte.gz", "data/train-images-idx3-ubyte.gz").ToList();

        var n = data.Count;

        var input = new double[n][];
        var output = new double[n][];

        for (int i = 0; i < n; i++)
        {
            var item = data[i];
            input[i] = ConvertByteArrayToDoubleArray(item.Image);
            output[i] = new double[10];
            output[i][item.Label] = 1.0;
        }

        return (input, output);
    }

    private static double[] ConvertByteArrayToDoubleArray(byte[,] byteArray)
    {
        // Get dimensions of the 2D array
        int rows = byteArray.GetLength(0);
        int cols = byteArray.GetLength(1);

        // Create the target double array
        double[] doubleArray = new double[rows * cols];

        // Copy each element, converting from byte to double
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                doubleArray[i * cols + j] = (double)byteArray[i, j] / 255.0; // Normalize to [0, 1]

        return doubleArray;
    }
}