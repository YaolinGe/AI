namespace OnnxValidator;
using System;
using System.Collections.Generic;
using System.Linq;

public class Program
{
    public static void Main(string[] args)
    {
        string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator";
        string dataPath = Path.Combine(rootFolder, "data.csv");
        string minMaxPath = Path.Combine(rootFolder, "min_max_values.csv");

        // Create FileHandler instance
        FileHandler fileHandler = new ();

        // Load data
        double[,] data = fileHandler.LoadData(dataPath);
        Dictionary<int, (double min, double max)> customRanges = fileHandler.LoadMinMaxValues(minMaxPath);

        // Create MinMaxScaler instance
        MinMaxScaler scaler = new MinMaxScaler();
        double[,] scaled = scaler.Transform(data, customRanges);

        // // Calculate the first difference
        //double[,] differenced = ComputeFirstDifference(scaled);

        //DataProcessor processor = new ();
    }
}
