namespace OnnxValidator;
using System;
using System.Collections.Generic;
using System.Linq;

public class Program
{
    public static void Main(string[] args)
    {
        // Load data into a 2D array
        string[] csvLines = File.ReadAllLines(@"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\data.csv");
        int rows = csvLines.Length - 1;
        int cols = csvLines[0].Split(',').Length;
        double[,] data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            double[] values = csvLines[i + 1].Split(',').Select(double.Parse).ToArray();
            for (int j = 0; j < cols; j++)
            {
                data[i, j] = values[j];
            }
        }

        // Load min max values into a dictionary 
        //string[] lines = File.ReadAllLines(@"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\min_max_values.csv");
        //Dictionary<int, (double min, double max)> customRanges = new ();
        //for (int i = 0; i < lines.Length - 1; i++)
        //{
        //    string[] values = lines[i + 1].Split(',');
        //    customRanges[i] = (double.Parse(values[0]), double.Parse(values[1]));
        //}

        //// Create MinMaxScaler instance
        //MinMaxScaler scaler = new MinMaxScaler();
        //double[,] scaled = scaler.Transform(data, customRanges);

        // Calculate the first difference
        //double[,] differenced = ComputeFirstDifference(scaled);

        DataProcessor processor = new ();
        //double[,] result = processor.Process(data);
    }
}
