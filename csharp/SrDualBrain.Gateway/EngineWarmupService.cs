using System.Text.Json.Nodes;

namespace SrDualBrain.Gateway;

public sealed class EngineWarmupService : BackgroundService
{
    private readonly PythonEngineClient _engine;
    private readonly ILogger<EngineWarmupService> _logger;

    public EngineWarmupService(PythonEngineClient engine, ILogger<EngineWarmupService> logger)
    {
        _engine = engine;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var enabled = (Environment.GetEnvironmentVariable("DUALBRAIN_ENGINE_AUTOSTART") ?? "1").Trim().ToLowerInvariant();
        if (enabled is "0" or "false" or "off" or "no")
        {
            _logger.LogInformation("Engine warm-up disabled (DUALBRAIN_ENGINE_AUTOSTART={Value})", enabled);
            return;
        }

        try
        {
            // Avoid competing with app startup; warm up shortly after the server starts.
            await Task.Delay(TimeSpan.FromMilliseconds(250), stoppingToken);

            await _engine.CallAsync(
                method: "health",
                @params: new JsonObject(),
                timeout: TimeSpan.FromSeconds(8),
                cancellationToken: stoppingToken);

            _logger.LogInformation("Python engine warm-up complete.");
        }
        catch (OperationCanceledException)
        {
            // normal shutdown
        }
        catch (Exception ex)
        {
            // Keep the gateway alive even if Python isn't ready yet.
            _logger.LogWarning(ex, "Python engine warm-up failed (gateway will continue).");
        }
    }
}

