// DataProcessor.cs
using OnnxValidator;
using System;
using System.Collections.Generic;
using System.Linq;

public class DataProcessor
{
    private readonly int[] _classicalLags = { 5, 10 };
    private readonly int _movingAverageWindow = 30;
    FileHandler fileHandler = new();
    MinMaxScaler scaler = new MinMaxScaler(0, 1);
    Dictionary<int, (double min, double max)> customRanges;

    public DataProcessor()
    {
        string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator";
        string minMaxPath = Path.Combine(rootFolder, "min_max_values.csv");
        customRanges = fileHandler.LoadMinMaxValues(minMaxPath);

    }

    public double[,] GetClassicalModelInput(double[,] data)
    {
        // s1, get preprocessed data
        double[,] processed = PreProcessInputData(data);

        // s2, create lags
        double[,] lagged = CreateLags(processed, _classicalLags);

        // s3, create moving averages
        double[,] movingAverage = CreateMovingAverage(processed, _movingAverageWindow);

        // s4, concatenate lagged and moving averages
        double[,] input = new double[processed.GetLength(0), processed.GetLength(1) + lagged.GetLength(1) + movingAverage.GetLength(1)];
        for (int i = 0; i < processed.GetLength(0); i++)
        {
            for (int j = 0; j < processed.GetLength(1); j++)
            {
                input[i, j] = processed[i, j];
            }
            for (int j = 0; j < lagged.GetLength(1); j++)
            {
                input[i, j + processed.GetLength(1)] = lagged[i, j];
            }
            for (int j = 0; j < movingAverage.GetLength(1); j++)
            {
                input[i, j + lagged.GetLength(1) + processed.GetLength(1)] = movingAverage[i, j];
            }
        }

        // s5, remove NaNs
        double[,] cleaned = RemoveNaNInPlace(input);

        return cleaned;
    }

    //public double[,] GetLSTMModelInput(double[,] data)
    //{
    //    double[,] processed = PreProcessInputData(data);

    //    double[,,] lstmInput = CreateLSTMSequence(processed, 30);
    //    return null;
    //}

    public double[,] PreProcessInputData(double[,] data)
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
        int totalNewCols = lags.Length * cols;
        double[,] result = new double[rows, totalNewCols];

        int currentCol = 0;
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

    public static double[,] CreateMovingAverage(double[,] data, int window)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        double[,] result = new double[rows, cols];

        int currentCol = 0;
        foreach (int col in Enumerable.Range(0, cols))
        {
            for (int i = 0; i < rows; i++)
            {
                double sum = 0;
                int count = 0;
                for (int j = i - window + 1; j <= i; j++)
                {
                    if (j < 0)
                        break; // Skip the first few rows
                    else
                    {
                        sum += data[j, col];
                        count++;
                    }
                }

                result[i, currentCol] = (count == 0) ? double.NaN : sum / count;
            }
            currentCol++;
        }

        return result;
    }

    public static double[,,] CreateLSTMSequence(double[,] data, int sequenceLength)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        int numSequences = rows - sequenceLength + 1;

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

    public static double[,] RemoveNaNInPlace(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        List<int> rowsToRemove = new List<int>();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (double.IsNaN(data[i, j]))
                {
                    rowsToRemove.Add(i);
                    break;
                }
            }
        }

        int newRows = rows - rowsToRemove.Count;
        double[,] cleanedData = new double[newRows, cols];
        int newRowIdx = 0;

        for (int i = 0; i < rows; i++)
        {
            if (!rowsToRemove.Contains(i))
            {
                for (int j = 0; j < cols; j++)
                {
                    cleanedData[newRowIdx, j] = data[i, j];
                }
                newRowIdx++;
            }
        }

        return cleanedData;
    }
}
