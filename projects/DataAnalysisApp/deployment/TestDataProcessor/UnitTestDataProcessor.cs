using Xunit;
using OnnxValidator;
using System.Collections.Generic;

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

        [Fact]
        public void TestCreateLags()
        {
            System.Diagnostics.Debug.WriteLine("TestCreateLags");
            // Arrange
            int[] lags = { 1, 2 };
            double[,] truth = new double[,]
            {
                { 1, 11, 21, 31, 41, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN },
                { 2, 12, 22, 32, 42, 1, double.NaN, 11, double.NaN, 21, double.NaN, 31, double.NaN, 41, double.NaN },
                { 3, 13, 23, 33, 43, 2, 1, 12, 11, 22, 21, 32, 31, 42, 41 },
                { 4, 14, 24, 34, 44, 3, 2, 13, 12, 23, 22, 33, 32, 43, 42 },
                { 5, 15, 25, 35, 45, 4, 3, 14, 13, 24, 23, 34, 33, 44, 43 },
                { 6, 16, 26, 36, 46, 5, 4, 15, 14, 25, 24, 35, 34, 45, 44 },
                { 7, 17, 27, 37, 47, 6, 5, 16, 15, 26, 25, 36, 35, 46, 45 },
                { 8, 18, 28, 38, 48, 7, 6, 17, 16, 27, 26, 37, 36, 47, 46 },
                { 9, 19, 29, 39, 49, 8, 7, 18, 17, 28, 27, 38, 37, 48, 47 },
                { 10, 20, 30, 40, 50, 9, 8, 19, 18, 29, 28, 39, 38, 49, 48 }
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
    }
}
