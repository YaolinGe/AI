using Xunit;
using OnnxValidator;
using System.Collections.Generic;

namespace TestDataProcessor
{
    public class UnitTestMinMaxScaler
    {
        private double[,] data = new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
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
            var scaler = new MinMaxScaler();
            double[,] truth = new double[,]
            {
                { 0, 0, 0 },
                { .5, .5, .5 },
                { 1, 1, 1 }
            };

            Dictionary<int, (double min, double max)> targetRange = new()
            {
                { 0, (0, 1) },
                { 1, (0, 1) },
                { 2, (0, 1) },
            };

            // Act
            double[,] transformed = scaler.Transform(data, targetRange);

            // Assert
            AssertTransformation(truth, transformed);
        }

        [Fact]
        public void TestFitTransform_Range_0_100()
        {
            System.Diagnostics.Debug.WriteLine("TestFitTransform_Range_0_100");

            // Arrange
            var scaler = new MinMaxScaler();
            double[,] truth = new double[,]
            {
                { 0, 0, 0 },
                { 50, 50, 50 },
                { 100, 100, 100 }
            };

            Dictionary<int, (double min, double max)> targetRange = new()
            {
                { 0, (0, 100) },
                { 1, (0, 100) },
                { 2, (0, 100) },
            };

            // Act
            double[,] transformed = scaler.Transform(data, targetRange);

            // Assert
            AssertTransformation(truth, transformed);
        }
    }
}
