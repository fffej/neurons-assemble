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

    public Vector(double[] elements, bool takeOwnership)
    {
        if (takeOwnership)
        {
            _elements = elements;
        }
        else
        {
            _elements = new double[elements.Length];
            Array.Copy(elements, _elements, elements.Length);
        }
    }

    public Vector(Vector vector, bool takeOwnership)
    {
        if (takeOwnership)
        {
            _elements = vector._elements;
        }
        else
        {
            _elements = new double[vector._elements.Length];
            Array.Copy(vector._elements, _elements, vector._elements.Length);
        }
    }    

    public double this[int index]
    {
        get { return _elements[index]; }
        set { _elements[index] = value; }
    }

    public int Length => _elements.Length;

    // In-place operations to avoid allocations
    public void CopyFrom(Vector source)
    {
        if (source.Length != Length)
            throw new ArgumentException("Vectors must be the same size for copy");
        Array.Copy(source._elements, _elements, Length);
    }

    public void CopyFrom(double[] source)
    {
        if (source.Length != Length)
            throw new ArgumentException("Array must be the same size for copy");
        Array.Copy(source, _elements, Length);
    }

    public void Zero()
    {
        Array.Clear(_elements);
    }

    public void AddInPlace(Vector other)
    {
        if (other.Length != Length)
            throw new ArgumentException("Vectors must be the same length for addition");
        
        for (int i = 0; i < Length; i++)
        {
            _elements[i] += other._elements[i];
        }
    }

    // In-place element-wise multiplication
    public void MultiplyInPlace(Vector other)
    {
        if (other.Length != Length)
            throw new ArgumentException("Vectors must be the same length for element-wise multiplication");
        
        for (int i = 0; i < Length; i++)
        {
            _elements[i] *= other._elements[i];
        }
    }

    // In-place scalar multiplication
    public void MultiplyInPlace(double scalar)
    {
        for (int i = 0; i < Length; i++)
        {
            _elements[i] *= scalar;
        }
    }

    // In-place scalar addition
    public void AddInPlace(double scalar)
    {
        for (int i = 0; i < Length; i++)
        {
            _elements[i] += scalar;
        }
    }

    public static double DotProduct(Vector v1, Vector v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Vectors must be the same size for dot product");

        double sum = 0;
        for (int i=0;i<v1.Length;i++)
            sum += v1[i] * v2[i];

        return sum;
    }
    
    public static Vector operator *(double scalar, Vector vector)
    {
        double[] result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = vector[i] * scalar;
        }
        return new Vector(result, true); // Take ownership
    }

    public static Vector operator *(Vector left, Vector right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must be the same length for element-wise multiplication");
            
        double[] result = new double[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = left[i] * right[i];
        }
        return new Vector(result, true); // Take ownership
    }

    // Vector addition - optimized
    public static Vector operator +(Vector left, Vector right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Vectors must be the same length for addition");
            
        double[] result = new double[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = left[i] + right[i];
        }
        return new Vector(result, true); // Take ownership
    }
    
    public override string ToString()
    {
        return string.Join(", ", _elements.Select((w, i) => $"W{i + 1}: {w:F6}"));
    }
}
