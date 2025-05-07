using System.Numerics;

namespace BackPropNN;

// We don't use anything built in, so that we can understand all the things that are happening!
class Vector
{
    private double[] _elements;

    public Vector(int size)
    {
        _elements = new double[size];
    }

    public Vector(double[] elements)
    {
        _elements = new double[elements.Length];
        Array.Copy(elements, _elements, elements.Length);
    }

    public double this[int index]
    {
        get { return _elements[index]; }
        set { _elements[index] = value; }
    }

    public int Length => _elements.Length;

    public static double DotProduct(Vector v1, Vector v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Vectors must be the same size for dot product");

        int i = 0;
        double sum = 0;

        // Process in chunks of Vector<double>.Count
        if (System.Numerics.Vector.IsHardwareAccelerated && v1.Length >= Vector<double>.Count)
        {
            int vectorSize = Vector<double>.Count;
            int lastVector = v1.Length - (v1.Length % vectorSize);
            Vector<double> sumVector = Vector<double>.Zero;

            for (; i < lastVector; i += vectorSize)
            {
                Vector<double> va = new Vector<double>(v1._elements, i);
                Vector<double> vb = new Vector<double>(v2._elements, i);
                sumVector += va * vb;
            }

            // Sum all elements in the vector
            for (int j = 0; j < vectorSize; j++)
            {
                sum += sumVector[j];
            }
        }

        // Process remaining elements
        for (; i < v1.Length; i++)
        {
            sum += v1[i] * v2[i];
        }

        return sum;
    }

    public override string ToString()
    {
        return string.Join(", ", _elements.Select((w, i) => $"W{i + 1}: {w:F6}"));
    }
}