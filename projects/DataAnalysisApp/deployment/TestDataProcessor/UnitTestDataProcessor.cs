using Xunit;
using OnnxValidator;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace TestDataProcessor
{
    public class UnitTestDataProcessor
    {
        private double[,] data = new double[,]
        {
            { 1, 11, 21, 31, 41 },
            { 2, 12, 22, 32, 42 },
            { 3, 13, 23, 33, 43 },
            { 4, 14, 24, 34, 44 },
            { 5, 15, 25, 35, 45 },
            { 6, 16, 26, 36, 46 },
            { 7, 17, 27, 37, 47 },
            { 8, 18, 28, 38, 48 },
            { 9, 19, 29, 39, 49 },
            { 10, 20, 30, 40, 50 }
        };

        private double[,] CalculateError(double[,] expected, double[,] actual)
        {
            int rows = expected.GetLength(0);
            int cols = expected.GetLength(1);
            double[,] errorArray = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    errorArray[i, j] = Math.Abs(expected[i, j] - actual[i, j]);
                }
            }

            return errorArray;
        }

        [Fact]
        public void TestCreateLags()
        {
            System.Diagnostics.Debug.WriteLine("TestCreateLags");
            // Arrange
            int[] lags = { 1, 2 };
            double[,] truth = new double[,]
            {
                { double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN },
                { 1, double.NaN, 11, double.NaN, 21, double.NaN, 31, double.NaN, 41, double.NaN },
                { 2, 1, 12, 11, 22, 21, 32, 31, 42, 41 },
                { 3, 2, 13, 12, 23, 22, 33, 32, 43, 42},
                { 4, 3, 14, 13, 24, 23, 34, 33, 44, 43},
                { 5, 4, 15, 14, 25, 24, 35, 34, 45, 44},
                { 6, 5, 16, 15, 26, 25, 36, 35, 46, 45},
                { 7, 6, 17, 16, 27, 26, 37, 36, 47, 46},
                { 8, 7, 18, 17, 28, 27, 38, 37, 48, 47},
                { 9, 8, 19, 18, 29, 28, 39, 38, 49, 48}
            };
            // Act
            double[,] lagsData = DataProcessor.CreateLags(data, lags);

            for (int i = 0; i < lagsData.GetLength(0); i++)
            {
                for (int j = 0; j < lagsData.GetLength(1); j++)
                {
                    System.Diagnostics.Debug.Write($"{lagsData[i, j]} ");
                }
                System.Diagnostics.Debug.WriteLine("");
            }

            // Assert
            Assert.Equal(truth, lagsData);
        }

        [Fact]
        public void TestCreateMovingAverage()
        {
            double[,] data = new double[,]
            {
                { 1, 5, 9 },
                { 2, 6, 10 },
                { 3, 7, 11 },
                { 4, 8, 12 }
            };

            int window = 2;

            double[,] truth = new double[,]
            {
                { double.NaN, double.NaN, double.NaN },
                { 1.5, 5.5, 9.5 },
                { 2.5, 6.5, 10.5 },
                { 3.5, 7.5, 11.5 },
            };

            double[,] movingAverage = DataProcessor.CreateMovingAverage(data, window);

            for (int i = 0; i < movingAverage.GetLength(0); i++)
            {
                for (int j = 0; j < movingAverage.GetLength(1); j++)
                {
                    System.Diagnostics.Debug.Write($"{movingAverage[i, j]} ");
                }
                System.Diagnostics.Debug.WriteLine("");
            }
            Assert.Equal(truth, movingAverage);
        }

        [Fact]
        public void TestCreateLSTMSequence()
        {
            double[,] data = new double[,]
            {
                { 1, 5, 9 },
                { 2, 6, 10 },
                { 3, 7, 11 },
                { 4, 8, 12 }
            };

            int sequenceLength = 2;

            double[,,] truth = new double[,,]
            {
                {
                    { 1, 5, 9 },
                    { 2, 6, 10 }
                },
                {
                    { 2, 6, 10 },
                    { 3, 7, 11 }
                },
                {
                    { 3, 7, 11 },
                    { 4, 8, 12 }
                }
            };

            double[,,] lstmSequence = DataProcessor.CreateLSTMSequence(data, sequenceLength);

            for (int i = 0; i < lstmSequence.GetLength(0); i++)
            {
                for (int j = 0; j < lstmSequence.GetLength(1); j++)
                {
                    for (int k = 0; k < lstmSequence.GetLength(2); k++)
                    {
                        System.Diagnostics.Debug.Write($"{lstmSequence[i, j, k]} ");
                    }
                    System.Diagnostics.Debug.WriteLine("");
                }
                System.Diagnostics.Debug.WriteLine("");
                System.Diagnostics.Debug.WriteLine("");
            }

            Assert.Equal(truth, lstmSequence);
        }

        [Fact]
        public void TestRemoveNaNInPlace()
        {
            double[,] data = new double[,]
            {
                { 1, 5, 9 },
                { 2, 6, 10 },
                { double.NaN, 7, 11 },
                { 4, 8, 12 }
            };
            double[,] truth = new double[,]
            {
                { 1, 5, 9 },
                { 2, 6, 10 },
                { 4, 8, 12 }
            };
            double[,] cleanedData = DataProcessor.RemoveNaNInPlace(data);
            for (int i = 0; i < cleanedData.GetLength(0); i++)
            {
                for (int j = 0; j < cleanedData.GetLength(1); j++)
                {
                    System.Diagnostics.Debug.Write($"{cleanedData[i, j]} ");
                }
                System.Diagnostics.Debug.WriteLine("");
            }
            Assert.Equal(truth, cleanedData);
        }

        [Fact]
        public void TestGetClassicalModelInput()
        {
            const double TOLERANCE = 1e-10; 
            string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator";
            string dataPath = Path.Combine(rootFolder, "data.csv");
            string truthPath = Path.Combine(rootFolder, "output.csv");  // ground truth from Python

            FileHandler fileHandler = new();
            double[,] data = fileHandler.LoadData(dataPath);
            double[,] truth = fileHandler.LoadData(truthPath);

            DataProcessor dataProcessor = new();
            double[,] classicalModelInput = dataProcessor.GetClassicalModelInput(data);
            double[,] errorArray = CalculateError(truth, classicalModelInput);

            for (int i = 0; i < errorArray.GetLength(0); i++)
            {
                for (int j = 0; j < errorArray.GetLength(1); j++)
                {
                    Assert.True(errorArray[i, j] < TOLERANCE, $"Error at ({i},{j}) is greater than tolerance.");
                }
            }
        }
    }
}
