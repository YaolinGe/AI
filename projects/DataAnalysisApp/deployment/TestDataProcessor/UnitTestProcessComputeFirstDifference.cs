using Xunit;
using OnnxValidator;
using System.Collections.Generic;

namespace TestDataProcessor
{
    public class ProcessComputeFirstDifference
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
        public void TestComputeFirstDifference()
        {
            System.Diagnostics.Debug.WriteLine("TestComputeFirstDifference");
            // Arrange

            double[,] truth = new double[,]
            {
                { 3, 3, 3 },
                { 3, 3, 3 }
            };

            // Act
            double[,] differenced = DataProcessor.ComputeFirstDifference(data);

            // Assert
            AssertTransformation(truth, differenced);
        }
    }
}
