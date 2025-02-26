using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using CsvHelper;
using System.Globalization;
using LiveDataViewer.Web.Models;

namespace LiveDataViewer.Web.Services;

public interface IDataStreamingService
{
    IObservable<List<SensorData>> DataStream { get; }
    StreamingConfiguration Configuration { get; }
    Task StartStreamingAsync(string filePath, CancellationToken cancellationToken);
    void StopStreaming();
    void UpdateStreamingSpeed(int speedMs);
    void UpdateWindowSize(int size);
}

public class DataStreamingService : IDataStreamingService
{
    private readonly Subject<List<SensorData>> _dataSubject = new();
    private CancellationTokenSource? _streamingCts;
    private readonly Queue<SensorData> _dataWindow = new();
    private readonly object _lock = new();
    private static readonly string[] NumericColumns = new[]
    {
        "x", "y", "z",
        "strain0", "strain1",
        "load", "deflection",
        "surfacefinish", "vibration"
    };

    public IObservable<List<SensorData>> DataStream => _dataSubject;
    public StreamingConfiguration Configuration { get; } = new();

    public async Task StartStreamingAsync(string filePath, CancellationToken cancellationToken)
    {
        if (Configuration.IsStreaming)
            return;

        Configuration.IsStreaming = true;
        _streamingCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

        try
        {
            await Task.Run(async () =>
            {
                using var reader = new StreamReader(filePath);
                using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);

                // Read headers
                csv.Read();
                csv.ReadHeader();

                while (!_streamingCts.Token.IsCancellationRequested && csv.Read())
                {
                    var timestampStr = csv.GetField("timestamp");
                    var sensorData = new SensorData 
                    { 
                        Timestamp = TimeSpan.Parse(timestampStr),
                        InCut = csv.GetField<int>("InCut") == 1,
                        Anomaly = csv.GetField<int>("Anomaly") == 1
                    };

                    foreach (var column in NumericColumns)
                    {
                        if (csv.TryGetField<double>(column, out var value))
                        {
                            sensorData.SensorValues[column] = value;
                        }
                    }

                    lock (_lock)
                    {
                        _dataWindow.Enqueue(sensorData);
                        if (_dataWindow.Count > Configuration.WindowSize)
                        {
                            _dataWindow.Dequeue();
                        }
                        _dataSubject.OnNext(_dataWindow.ToList());
                    }

                    await Task.Delay(Configuration.StreamingSpeedMs, _streamingCts.Token);
                }
            }, _streamingCts.Token);
        }
        catch (OperationCanceledException)
        {
            // Normal cancellation, ignore
        }
        finally
        {
            Configuration.IsStreaming = false;
        }
    }

    public void StopStreaming()
    {
        _streamingCts?.Cancel();
        Configuration.IsStreaming = false;
    }

    public void UpdateStreamingSpeed(int speedMs)
    {
        Configuration.StreamingSpeedMs = Math.Max(10, speedMs);
    }

    public void UpdateWindowSize(int size)
    {
        lock (_lock)
        {
            Configuration.WindowSize = Math.Max(10, size);
            while (_dataWindow.Count > Configuration.WindowSize)
            {
                _dataWindow.Dequeue();
            }
            if (_dataWindow.Any())
            {
                _dataSubject.OnNext(_dataWindow.ToList());
            }
        }
    }
} 