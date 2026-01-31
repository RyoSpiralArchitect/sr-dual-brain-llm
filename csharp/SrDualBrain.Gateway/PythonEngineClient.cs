using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace SrDualBrain.Gateway;

public sealed class PythonEngineClient : IAsyncDisposable
{
    private readonly ILogger<PythonEngineClient> _logger;
    private readonly string _pythonExe;
    private readonly string _engineScriptPath;
    private readonly string _workingDirectory;
    private readonly TimeSpan _defaultTimeout;
    private readonly SemaphoreSlim _writeLock = new(1, 1);

    private readonly ConcurrentDictionary<string, TaskCompletionSource<JsonObject>> _pending = new();
    private readonly CancellationTokenSource _cts = new();

    private Process? _process;
    private Task? _stdoutPump;
    private Task? _stderrPump;

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
    };

    public PythonEngineClient(
        ILogger<PythonEngineClient> logger,
        string pythonExe,
        string engineScriptPath,
        string workingDirectory,
        TimeSpan defaultTimeout)
    {
        _logger = logger;
        _pythonExe = pythonExe;
        _engineScriptPath = engineScriptPath;
        _workingDirectory = workingDirectory;
        _defaultTimeout = defaultTimeout;
    }

    public async Task<JsonObject> CallAsync(string method, JsonObject? @params, CancellationToken cancellationToken)
    {
        return await CallAsync(method, @params, timeout: _defaultTimeout, cancellationToken);
    }

    public async Task<JsonObject> CallAsync(string method, JsonObject? @params, TimeSpan timeout, CancellationToken cancellationToken)
    {
        EnsureStarted();

        var id = Guid.NewGuid().ToString("N");
        var request = new JsonObject
        {
            ["id"] = id,
            ["method"] = method,
            ["params"] = @params ?? new JsonObject(),
        };

        var tcs = new TaskCompletionSource<JsonObject>(TaskCreationOptions.RunContinuationsAsynchronously);
        if (!_pending.TryAdd(id, tcs))
        {
            throw new InvalidOperationException("Failed to enqueue request.");
        }

        try
        {
            var line = JsonSerializer.Serialize(request, JsonOptions);
            await WriteLineAsync(line, cancellationToken);

            var response = await tcs.Task.WaitAsync(timeout, cancellationToken);
            var ok = response["ok"]?.GetValue<bool>() ?? false;
            if (!ok)
            {
                var err = response["error"] as JsonObject;
                var errType = err?["type"]?.GetValue<string>() ?? "EngineError";
                var errMsg = err?["message"]?.GetValue<string>() ?? "unknown error";
                throw new InvalidOperationException($"{errType}: {errMsg}");
            }

            var result = response["result"] as JsonObject;
            // Detach from the response tree (JsonNode instances cannot be re-parented).
            return (result?.DeepClone() as JsonObject) ?? new JsonObject();
        }
        finally
        {
            _pending.TryRemove(id, out _);
        }
    }

    private async Task WriteLineAsync(string line, CancellationToken cancellationToken)
    {
        var proc = _process;
        if (proc is null || proc.HasExited)
        {
            throw new InvalidOperationException("Python engine is not running.");
        }

        await _writeLock.WaitAsync(cancellationToken);
        try
        {
            await proc.StandardInput.WriteLineAsync(line.AsMemory(), cancellationToken);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private void EnsureStarted()
    {
        if (_process is { HasExited: false })
        {
            return;
        }

        StartProcess();
    }

    private void StartProcess()
    {
        FailAllPending(new InvalidOperationException("Python engine restarted."));

        var startInfo = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = $"-u \"{_engineScriptPath}\"",
            WorkingDirectory = _workingDirectory,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        startInfo.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";

        _process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        _process.Exited += (_, _) =>
        {
            _logger.LogError("Python engine exited with code {ExitCode}", _process.ExitCode);
            FailAllPending(new InvalidOperationException("Python engine exited."));
        };

        if (!_process.Start())
        {
            throw new InvalidOperationException("Failed to start python engine.");
        }
        _process.StandardInput.AutoFlush = true;
        _process.StandardInput.NewLine = "\n";

        _stdoutPump = Task.Run(() => PumpStdoutAsync(_process, _cts.Token));
        _stderrPump = Task.Run(() => PumpStderrAsync(_process, _cts.Token));

        _logger.LogInformation("Python engine started (pid={Pid})", _process.Id);
    }

    private async Task PumpStdoutAsync(Process process, CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested && !process.HasExited)
            {
                var line = await process.StandardOutput.ReadLineAsync(cancellationToken);
                if (line is null)
                {
                    break;
                }

                if (!line.TrimStart().StartsWith("{", StringComparison.Ordinal))
                {
                    // Some environments inject startup banners to stdout; ignore them.
                    _logger.LogDebug("engine stdout: {Line}", line);
                    continue;
                }

                JsonObject? obj;
                try
                {
                    obj = JsonSerializer.Deserialize<JsonObject>(line, JsonOptions);
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "Failed to parse engine response line: {Line}", line);
                    continue;
                }

                var id = obj?["id"]?.GetValue<string>();
                if (id is null)
                {
                    continue;
                }

                if (_pending.TryGetValue(id, out var tcs))
                {
                    tcs.TrySetResult(obj!);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // normal shutdown
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Engine stdout pump failed.");
            FailAllPending(ex);
        }
    }

    private async Task PumpStderrAsync(Process process, CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested && !process.HasExited)
            {
                var line = await process.StandardError.ReadLineAsync(cancellationToken);
                if (line is null)
                {
                    break;
                }

                _logger.LogWarning("engine: {Line}", line);
            }
        }
        catch (OperationCanceledException)
        {
            // normal shutdown
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Engine stderr pump failed.");
        }
    }

    private void FailAllPending(Exception ex)
    {
        foreach (var item in _pending)
        {
            item.Value.TrySetException(ex);
        }
        _pending.Clear();
    }

    public async ValueTask DisposeAsync()
    {
        _cts.Cancel();

        try
        {
            if (_stdoutPump is not null)
            {
                await _stdoutPump;
            }
        }
        catch
        {
            // ignore
        }

        try
        {
            if (_stderrPump is not null)
            {
                await _stderrPump;
            }
        }
        catch
        {
            // ignore
        }

        if (_process is not null)
        {
            try
            {
                if (!_process.HasExited)
                {
                    _process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // ignore
            }
            _process.Dispose();
        }

        _cts.Dispose();
        _writeLock.Dispose();
    }
}
