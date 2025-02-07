// DataProcessor.cs
using System;
using System.Collections.Generic;
using System.Linq;

public class DataProcessor
{
    private readonly int[] _classicalLags = { 5, 10 };
    private readonly int[] _windows = { 30 };


    public double[,] Process(double[,] data)
    {
        // Step 1: Select and scale raw columns
        var scaled = new MinMaxScaler().FitTransform(data);

        // Step 2: Compute first difference
        var differenced = ComputeFirstDifference(scaled);

        // Step 3: Clean data
        var cleaned = DropNA(differenced);

        // Step 4: Split data
        var (train, val, test) = SplitData(cleaned);

        // Step 5: Create features
        //return new ProcessedData(
        //    CreateClassicalFeatures(train, val, test),
        //    CreateLSTMFeatures(cleaned, train.GetLength(0), val.GetLength(0), test.GetLength(0))
        //);
        return new double[0, 0];
    }

    static double[,] ComputeFirstDifference(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);

        if (rows < 2)
            throw new ArgumentException("Input matrix must have at least two rows.");

        double[,] diff = new double[rows - 1, cols];

        for (int i = 1; i < rows; i++)  // Start from the second row
        {
            for (int j = 0; j < cols; j++)
            {
                diff[i - 1, j] = data[i, j] - data[i - 1, j];
            }
        }

        return diff;
    }

    static double[,] DropNA(double[,] input)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        List<double[]> cleanedData = new List<double[]>();

        for (int i = 0; i < rows; i++)
        {
            bool hasNaN = false;
            for (int j = 0; j < cols; j++)
            {
                if (double.IsNaN(input[i, j]))
                {
                    hasNaN = true;
                    break;
                }
            }
            if (!hasNaN)
            {
                double[] row = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    row[j] = input[i, j];
                }
                cleanedData.Add(row);
            }
        }

        double[,] cleanedArray = new double[cleanedData.Count, cols];
        for (int i = 0; i < cleanedData.Count; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                cleanedArray[i, j] = cleanedData[i][j];
            }
        }

        return cleanedArray;
    }

    private (double[,], double[,], double[,]) SplitData(double[,] data)
    {
        int rowCount = data.GetLength(0);
        int trainEnd = (int)(rowCount * 0.6);
        int valEnd = trainEnd + (int)(rowCount * 0.2);

        return (
            Slice(data, 0, trainEnd),
            Slice(data, trainEnd, valEnd),
            Slice(data, valEnd, rowCount)
        );
    }

    private double[,] Slice(double[,] data, int start, int end)
    {
        int cols = data.GetLength(1);
        double[,] slice = new double[end - start, cols];

        for (int i = start; i < end; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                slice[i - start, j] = data[i, j];
            }
        }

        return slice;
    }

    private double[,] CreateClassicalFeatures(double[,] train, double[,] val, double[,] test)
    {
        var trainFeatures = AddFeatures(train);
        var valFeatures = AddFeatures(val);
        var testFeatures = AddFeatures(test);

        return CombineFeatures(trainFeatures, valFeatures, testFeatures);
    }

    private double[,] AddFeatures(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        List<double[]> enhancedData = new List<double[]>();

        for (int i = 0; i < rows; i++)
        {
            double[] row = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                row[j] = data[i, j];
            }
            enhancedData.Add(row);
        }

        // Add lag features
        foreach (var lag in _classicalLags)
        {
            for (int j = 0; j < cols; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    if (i >= lag)
                    {
                        enhancedData[i] = enhancedData[i].Concat(new double[] { data[i - lag, j] }).ToArray();
                    }
                    else
                    {
                        enhancedData[i] = enhancedData[i].Concat(new double[] { double.NaN }).ToArray();
                    }
                }
            }
        }

        // Add moving averages
        foreach (var window in _windows)
        {
            for (int j = 0; j < cols; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    var start = Math.Max(0, i - window + 1);
                    var ma = data.Skip(start).Take(i - start + 1).Select(row => row[j]).Average();
                    enhancedData[i] = enhancedData[i].Concat(new double[] { ma }).ToArray();
                }
            }
        }

        return DropNA(ToArray(enhancedData));
    }

    private double[,] ToArray(List<double[]> data)
    {
        int rows = data.Count;
        int cols = data[0].Length;
        double[,] array = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i, j] = data[i][j];
            }
        }

        return array;
    }

    private double[,] CombineFeatures(double[,] train, double[,] val, double[,] test)
    {
        int trainRows = train.GetLength(0);
        int valRows = val.GetLength(0);
        int testRows = test.GetLength(0);
        int cols = train.GetLength(1);

        double[,] combined = new double[trainRows + valRows + testRows, cols];

        for (int i = 0; i < trainRows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                combined[i, j] = train[i, j];
            }
        }

        for (int i = 0; i < valRows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                combined[trainRows + i, j] = val[i, j];
            }
        }

        for (int i = 0; i < testRows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                combined[trainRows + valRows + i, j] = test[i, j];
            }
        }

        return combined;
    }

    private double[,] CreateLSTMFeatures(double[,] data, int trainRows, int valRows, int testRows)
    {
        // Implement LSTM feature creation logic here
        return new double[0, 0];
    }
}
