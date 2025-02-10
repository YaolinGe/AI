// DataProcessor.cs
using OnnxValidator;
using System;
using System.Collections.Generic;
using System.Linq;

public class DataProcessor
{
    private readonly int[] _classicalLags = { 5, 10 };
    private readonly int[] _windows = { 30 };
    MinMaxScaler scaler = new MinMaxScaler(0, 1);
    Dictionary<int, (double min, double max)> customRanges = new();

    public DataProcessor()
    {

    }

    public double[,] Process(double[,] data)
    {
        // Step 1: Select and scale raw columns
        double[,] scaled = scaler.Transform(data, customRanges);

        // Step 2: Compute first difference
        double[,] differenced = ComputeFirstDifference(scaled);

        return differenced;
    }

    public static double[,] ComputeFirstDifference(double[,] data)
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

    public static double[,] CreateLags(double[,] data, int[] lags)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        int totalNewCols = cols + (lags.Length * cols);
        double[,] result = new double[rows, totalNewCols];

        // Copy original data
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = data[i, j];
            }
        }

        int currentCol = cols;
        foreach (int col in Enumerable.Range(0, cols))
        {
            foreach (int lag in lags)
            {
                for (int i = 0; i < rows; i++)
                {
                    result[i, currentCol] = (i - lag >= 0) ? data[i - lag, col] : double.NaN;
                }
                currentCol++;
            }
        }

        return result;
    }

    public static double[,] CreateMovingAverageAndStd(double[,] data, int[] windows, int[] columns)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        int totalNewCols = cols + (windows.Length * columns.Length);
        double[,] result = new double[rows, totalNewCols];

        // Copy original data
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = data[i, j];
            }
        }

        int currentCol = cols;
        foreach (int col in columns)
        {
            foreach (int window in windows)
            {
                for (int i = 0; i < rows; i++)
                {
                    double sum = 0;
                    int count = 0;

                    for (int j = Math.Max(0, i - window + 1); j <= i; j++)
                    {
                        sum += data[j, col];
                        count++;
                    }

                    result[i, currentCol] = (count > 0) ? sum / count : 0;
                }
                currentCol++;
            }
        }

        return result;
    }

    public static double[,,] CreateSequenceLSTM(double[,] data, int sequenceLength)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        int numSequences = rows - sequenceLength;

        // Create 3D array: [sequences, sequence_length, features]
        double[,,] sequences = new double[numSequences, sequenceLength, cols];

        for (int i = 0; i < numSequences; i++)
        {
            for (int j = 0; j < sequenceLength; j++)
            {
                for (int k = 0; k < cols; k++)
                {
                    sequences[i, j, k] = data[i + j, k];
                }
            }
        }

        return sequences;
    }
}
