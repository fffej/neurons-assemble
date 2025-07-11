using MNIST.IO;
using System;
using System.IO;

public class MNistLoader
{
    public static (double[][], double[][]) Load()
    {
        // Use AppContext.BaseDirectory to get the directory where the app is running
        string baseDir = AppContext.BaseDirectory;
        string dataDir = Path.Combine(baseDir, "data");
        string labelsPath = Path.Combine(dataDir, "train-labels-idx1-ubyte.gz");
        string imagesPath = Path.Combine(dataDir, "train-images-idx3-ubyte.gz");

        var data = FileReaderMNIST.LoadImagesAndLables(labelsPath, imagesPath).ToList();

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