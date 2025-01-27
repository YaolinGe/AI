// UnitTests.cs
using Xunit;
using System.Linq;

public class DataProcessorTests
{
    [Fact]
    public void Test_FullPipeline()
    {
        // Create test data
        var df = new DataInput();
        var rand = new Random();
        df.AddColumn("x", GenerateRandom(1000, rand));
        df.AddColumn("y", GenerateRandom(1000, rand));
        df.AddColumn("z", GenerateRandom(1000, rand));
        df.AddColumn("strain0", GenerateRandom(1000, rand));
        df.AddColumn("strain1", GenerateRandom(1000, rand));

        // Process data
        var processor = new DataProcessor();
        var processed = processor.Process(df);

        // Verify results
        Assert.NotNull(processed.Classical.Train);
        Assert.NotNull(processed.LSTM.Train);
        Assert.Equal(780, processed.Classical.Train.Length); // 1000 * 0.78 ≈ 780
        Assert.All(processed.LSTM.Train, seq => Assert.Equal(30, seq.Length));
    }

    private static List<double> GenerateRandom(int count, Random rand) =>
        Enumerable.Range(0, count).Select(_ => rand.NextDouble()).ToList();
}