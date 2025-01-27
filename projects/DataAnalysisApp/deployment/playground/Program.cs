using System;

class Program
{
    static void Main()
    {
        double[,] input = {
            {1, 2, 3},
            {5.5, 4, 6},
            {5.5, 6, 9},
            {10, 8, 12}
        };
        
        double[,] scaled = MinMaxScale(input); 
        double[,] difference = ComputeFirstDifference(scaled);

        
        // Print the result
        PrintMatrix(input);
        PrintMatrix(scaled);
        PrintMatrix(difference);
    }

    static double[,] ComputeFirstDifference(double[,] input)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);

        if (rows < 2)
            throw new ArgumentException("Input matrix must have at least two rows.");

        double[,] diff = new double[rows - 1, cols];

        for (int i = 1; i < rows; i++)  // Start from the second row
        {
            for (int j = 0; j < cols; j++)
            {
                diff[i - 1, j] = input[i, j] - input[i - 1, j];
            }
        }

        return diff;
    }
    
    
    static double[,] MinMaxScale(double[,] input)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);

        double[,] scaled = new double[rows, cols];

        for (int j = 0; j < cols; j++)  // Scale each column independently
        {
            double min = double.MaxValue;
            double max = double.MinValue;

            // Find min and max for the column
            for (int i = 0; i < rows; i++)
            {
                if (input[i, j] < min) min = input[i, j];
                if (input[i, j] > max) max = input[i, j];
            }

            double range = max - min;
            if (range == 0)
            {
                // If all values are the same, set everything to 0 (or handle differently)
                for (int i = 0; i < rows; i++)
                {
                    scaled[i, j] = 0;
                }
            }
            else
            {
                // Apply Min-Max Scaling
                for (int i = 0; i < rows; i++)
                {
                    scaled[i, j] = (input[i, j] - min) / range;
                }
            }
        }

        return scaled;
    }

    static void PrintMatrix(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Console.Write(matrix[i, j].ToString("F2") + "\t");
            }
            Console.WriteLine();
        }
    }
}