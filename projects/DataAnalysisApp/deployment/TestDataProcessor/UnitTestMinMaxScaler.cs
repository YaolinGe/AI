using Xunit;
using OnnxValidator;
using System.Collections.Generic;

namespace TestDataProcessor
{
    public class UnitTestMinMaxScaler
    {
        private double[,] data = new double[,]
        {
            { 1, 4, 7 },
            { 2, 5, 8 },
            { 3, 6, 9 }
        };

        private void AssertTransformation(double[,] expected, double[,] actual)
        {
            int rows = expected.GetLength(0);
            int cols = expected.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    Assert.Equal(expected[i, j], actual[i, j]);
                }
            }
        }

        [Fact]
        public void TestFitTransform_Range_0_1()
        {
            System.Diagnostics.Debug.WriteLine("TestFitTransform_Range_0_1");

            // Arrange
            MinMaxScaler scaler = new(0, 1);

            double[,] truth = new double[,]
            {
                { 0, 0, 0 },
                { 0.5, 0.5, 0.5 },
                { 1, 1, 1 }
            };

            Dictionary<int, (double min, double max)> dataRange = new()
            {
                { 0, (1, 3) },
                { 1, (4, 6) },
                { 2, (7, 9) }
            };

            // Act
            double[,] transformed = scaler.Transform(data, dataRange);

            // Assert
            AssertTransformation(truth, transformed);
        }

        [Fact]
        public void TestComputeFirstDifference()
        {
            System.Diagnostics.Debug.WriteLine("TestComputeFirstDifference");
            // Arrange

            double[,] truth = new double[,]
            {
                { 1, 1, 1 },
                { 1, 1, 1 }
            };

            // Act
            double[,] differenced = DataProcessor.ComputeFirstDifference(data);

            // Assert
            AssertTransformation(truth, differenced);
        }
    }
}
