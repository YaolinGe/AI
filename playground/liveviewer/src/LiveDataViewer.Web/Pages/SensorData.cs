using System;
using System.Collections.Generic;

namespace LiveDataViewer.Web.Pages;

public class SensorData
{
    public TimeSpan Timestamp { get; set; }
    public Dictionary<string, double> SensorValues { get; set; } = new();
    public bool InCut { get; set; }
    public bool Anomaly { get; set; }
}

public class StreamingConfiguration
{
    public int WindowSize { get; set; } = 100;
    public int StreamingSpeedMs { get; set; } = 1000;
    public bool IsStreaming { get; set; }
    
    public static readonly string[] SensorGroups = new[]
    {
        "Position (x,y,z)",
        "Strain (strain0, strain1)",
        "Process Parameters (load, deflection)",
        "Quality Metrics (surfacefinish, vibration)"
    };
} 