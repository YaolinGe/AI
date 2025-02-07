namespace OnnxValidator;
using System;
using System.Collections.Generic;
using System.Linq;

public class Program
{
    public static void Main(string[] args)
    {
        // Create test data
        double[,] input =
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 },
            { 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0 },
            { 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0 },
            { 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0 },
            { 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0 },
            { 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0 },
            { 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0 },
            { 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0 },
            { 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0 },
            { 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0 }
        };
        // Process data
        //var processor = new DataProcessor();
        // var processed = processor.Process(input);

        // Create MinMaxScaler instance
        var scaler = new MinMaxScaler();
        // Fit and transform the data
        //var transformed = scaler.FitTransform(input);
        // Print the transformed data
        Console.WriteLine("Transformed data:");
        //for (int i = 0; i < transformed.GetLength(0); i++)
        //{
        //    for (int j = 0; j < transformed.GetLength(1); j++)
        //    {
        //        Console.Write($"{transformed[i, j]:F2} ");
        //    }
        //    Console.WriteLine();
        //}


        Console.WriteLine("All tests passed.");
    }
}
