using System.Text.Json.Nodes;

namespace SrDualBrain.Gateway;

public sealed class EngineHealthCache
{
    private readonly PythonEngineClient _engine;
    private readonly ILogger<EngineHealthCache> _logger;
    private readonly SemaphoreSlim _refreshLock = new(1, 1);
    private readonly TimeSpan _ttl;

    private JsonObject? _cached;
    private DateTimeOffset _cachedAt = DateTimeOffset.MinValue;

    public EngineHealthCache(PythonEngineClient engine, ILogger<EngineHealthCache> logger)
    {
        _engine = engine;
        _logger = logger;
        _ttl = LoadTtl();
    }

    public async Task<JsonObject> GetAsync(CancellationToken cancellationToken)
    {
        var now = DateTimeOffset.UtcNow;
        var cached = _cached;
        if (cached is not null && now - _cachedAt <= _ttl)
        {
            return (cached.DeepClone() as JsonObject) ?? new JsonObject();
        }

        await _refreshLock.WaitAsync(cancellationToken);
        try
        {
            now = DateTimeOffset.UtcNow;
            cached = _cached;
            if (cached is not null && now - _cachedAt <= _ttl)
            {
                return (cached.DeepClone() as JsonObject) ?? new JsonObject();
            }

            var fresh = await _engine.CallAsync(
                "health",
                new JsonObject(),
                timeout: TimeSpan.FromSeconds(2),
                cancellationToken);
            _cached = fresh;
            _cachedAt = now;
            return (fresh.DeepClone() as JsonObject) ?? new JsonObject();
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Engine health refresh failed.");
            if (_cached is not null)
            {
                return (_cached.DeepClone() as JsonObject) ?? new JsonObject();
            }
            return new JsonObject
            {
                ["status"] = "error",
                ["error"] = ex.Message,
            };
        }
        finally
        {
            _refreshLock.Release();
        }
    }

    private static TimeSpan LoadTtl()
    {
        var raw = Environment.GetEnvironmentVariable("DUALBRAIN_ENGINE_HEALTH_TTL_MS");
        if (!int.TryParse(raw, out var ttlMs))
        {
            ttlMs = 1000;
        }
        ttlMs = Math.Clamp(ttlMs, 100, 60_000);
        return TimeSpan.FromMilliseconds(ttlMs);
    }
}

