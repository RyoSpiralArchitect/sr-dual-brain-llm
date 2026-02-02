using System.Collections.Concurrent;
using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using System.IO.Pipes;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;

namespace SrDualBrain.Gateway;

public sealed class PythonEngineClient : IAsyncDisposable
{
    private readonly ILogger<PythonEngineClient> _logger;
    private readonly string _pythonExe;
    private readonly string _engineScriptPath;
    private readonly string _workingDirectory;
    private readonly TimeSpan _defaultTimeout;
    private readonly string _transport;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly SemaphoreSlim _startLock = new(1, 1);
    private readonly SemaphoreSlim _blobLock = new(1, 1);

    private readonly ConcurrentDictionary<string, TaskCompletionSource<JsonObject>> _pending = new();
    private readonly ConcurrentDictionary<string, ChannelWriter<JsonObject>> _streams = new();
    private readonly CancellationTokenSource _cts = new();

    private Process? _process;
    private Task? _stdoutPump;
    private Task? _stderrPump;
    private Task? _pipePump;
    private NamedPipeServerStream? _pipe;
    private string? _pipeName;
    private NamedPipeServerStream? _blobPipe;
    private string? _blobPipeName;

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
    };

    public PythonEngineClient(
        ILogger<PythonEngineClient> logger,
        string pythonExe,
        string engineScriptPath,
        string workingDirectory,
        TimeSpan defaultTimeout,
        string? transport = null)
    {
        _logger = logger;
        _pythonExe = pythonExe;
        _engineScriptPath = engineScriptPath;
        _workingDirectory = workingDirectory;
        _defaultTimeout = defaultTimeout;
        _transport = (transport ?? "stdio").Trim();
    }

    public async Task<JsonObject> CallAsync(string method, JsonObject? @params, CancellationToken cancellationToken)
    {
        return await CallAsync(method, @params, timeout: _defaultTimeout, cancellationToken);
    }

    public async Task<JsonObject> CallAsync(string method, JsonObject? @params, TimeSpan timeout, CancellationToken cancellationToken)
    {
        await EnsureStartedAsync(cancellationToken);

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
            await WriteMessageAsync(line, cancellationToken);

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

    public async Task RestartAsync(CancellationToken cancellationToken)
    {
        await _startLock.WaitAsync(cancellationToken);
        try
        {
            try
            {
                if (_process is not null && !_process.HasExited)
                {
                    _process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // ignore
            }

            try
            {
                _process?.Dispose();
            }
            catch
            {
                // ignore
            }

            _process = null;
            StartProcess();
        }
        finally
        {
            _startLock.Release();
        }
    }

    public async IAsyncEnumerable<JsonObject> CallStreamAsync(
        string method,
        JsonObject? @params,
        TimeSpan timeout,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await EnsureStartedAsync(cancellationToken);

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

        var channel = Channel.CreateUnbounded<JsonObject>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = false,
        });
        if (!_streams.TryAdd(id, channel.Writer))
        {
            _pending.TryRemove(id, out _);
            throw new InvalidOperationException("Failed to register stream.");
        }

        try
        {
            var line = JsonSerializer.Serialize(request, JsonOptions);
            await WriteMessageAsync(line, cancellationToken);

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(timeout);

            while (true)
            {
                if (tcs.Task.IsCompleted)
                {
                    yield return await tcs.Task;
                    yield break;
                }

                var canRead = await channel.Reader.WaitToReadAsync(timeoutCts.Token);
                if (!canRead)
                {
                    // Stream finished; ensure final response is observed.
                    yield return await tcs.Task.WaitAsync(timeoutCts.Token);
                    yield break;
                }

                while (channel.Reader.TryRead(out var message))
                {
                    yield return message;
                }
            }
        }
        finally
        {
            _streams.TryRemove(id, out _);
            _pending.TryRemove(id, out _);
            try
            {
                channel.Writer.TryComplete();
            }
            catch
            {
                // ignore
            }
        }
    }

    public async IAsyncEnumerable<JsonObject> CallStreamAsync(
        string method,
        JsonObject? @params,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var ev in CallStreamAsync(method, @params, _defaultTimeout, cancellationToken))
        {
            yield return ev;
        }
    }

    public async Task<JsonObject> PutBlobAsync(
        string sessionId,
        Stream content,
        long length,
        string? contentType,
        string? fileName,
        CancellationToken cancellationToken)
    {
        if (!IsPipeTransport())
        {
            throw new InvalidOperationException("Blob uploads require DUALBRAIN_ENGINE_TRANSPORT=pipes.");
        }

        sessionId = string.IsNullOrWhiteSpace(sessionId) ? "default" : sessionId.Trim();
        if (length <= 0)
        {
            throw new InvalidOperationException("Blob length must be > 0.");
        }
        if (length > uint.MaxValue)
        {
            throw new InvalidOperationException($"Blob too large for framing: {length} bytes");
        }

        await _blobLock.WaitAsync(cancellationToken);
        try
        {
            await EnsureStartedAsync(cancellationToken);

            var blobPipe = _blobPipe;
            if (blobPipe is null || !blobPipe.IsConnected)
            {
                throw new InvalidOperationException("Blob pipe is not connected.");
            }

            var blobId = Guid.NewGuid().ToString("N");
            var header = new JsonObject
            {
                ["session_id"] = sessionId,
                ["blob_id"] = blobId,
                ["content_type"] = string.IsNullOrWhiteSpace(contentType) ? null : contentType,
                ["file_name"] = string.IsNullOrWhiteSpace(fileName) ? null : fileName,
            };

            var headerBytes = JsonSerializer.SerializeToUtf8Bytes(header, JsonOptions);

            var writeTask = WriteBlobFrameAsync(blobPipe, headerBytes, content, (uint)length, cancellationToken);

            var result = await CallAsync(
                "blob_put",
                new JsonObject
                {
                    ["session_id"] = sessionId,
                    ["blob_id"] = blobId,
                },
                cancellationToken);

            await writeTask;
            return result;
        }
        finally
        {
            _blobLock.Release();
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

    private static async Task WriteBlobFrameAsync(
        NamedPipeServerStream blobPipe,
        byte[] headerBytes,
        Stream content,
        uint contentLength,
        CancellationToken cancellationToken)
    {
        if (headerBytes.Length <= 0)
        {
            throw new InvalidOperationException("Blob header is empty.");
        }
        if (headerBytes.Length > 64 * 1024)
        {
            throw new InvalidOperationException($"Blob header too large: {headerBytes.Length} bytes");
        }

        var headerLenBuf = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(headerLenBuf, (uint)headerBytes.Length);
        var dataLenBuf = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(dataLenBuf, contentLength);

        await blobPipe.WriteAsync(headerLenBuf, cancellationToken);
        await blobPipe.WriteAsync(headerBytes, cancellationToken);
        await blobPipe.WriteAsync(dataLenBuf, cancellationToken);

        var remaining = (long)contentLength;
        var pool = ArrayPool<byte>.Shared;
        var buffer = pool.Rent(256 * 1024);
        try
        {
            while (remaining > 0)
            {
                var toRead = (int)Math.Min(buffer.Length, remaining);
                var read = await content.ReadAsync(buffer.AsMemory(0, toRead), cancellationToken);
                if (read <= 0)
                {
                    throw new EndOfStreamException("Unexpected end of blob content stream.");
                }
                await blobPipe.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                remaining -= read;
            }
        }
        finally
        {
            pool.Return(buffer);
        }

        await blobPipe.FlushAsync(cancellationToken);
    }

    private async Task WritePipeFrameAsync(string json, CancellationToken cancellationToken)
    {
        var pipe = _pipe;
        if (pipe is null || !pipe.IsConnected)
        {
            throw new InvalidOperationException("Python engine pipe is not connected.");
        }

        var payload = Encoding.UTF8.GetBytes(json);
        if (payload.Length <= 0)
        {
            return;
        }
        if (payload.Length > 64 * 1024 * 1024)
        {
            throw new InvalidOperationException($"Engine message too large: {payload.Length} bytes");
        }

        var header = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(header, payload.Length);

        await _writeLock.WaitAsync(cancellationToken);
        try
        {
            await pipe.WriteAsync(header, cancellationToken);
            await pipe.WriteAsync(payload, cancellationToken);
            await pipe.FlushAsync(cancellationToken);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private async Task WriteMessageAsync(string line, CancellationToken cancellationToken)
    {
        if (IsPipeTransport())
        {
            await WritePipeFrameAsync(line, cancellationToken);
            return;
        }
        await WriteLineAsync(line, cancellationToken);
    }

    private static async Task ReadExactAsync(Stream stream, Memory<byte> buffer, CancellationToken cancellationToken)
    {
        var offset = 0;
        while (offset < buffer.Length)
        {
            var read = await stream.ReadAsync(buffer.Slice(offset), cancellationToken);
            if (read <= 0)
            {
                throw new EndOfStreamException("Pipe closed");
            }
            offset += read;
        }
    }

    private async Task PumpPipeAsync(NamedPipeServerStream pipe, CancellationToken cancellationToken)
    {
        try
        {
            var header = new byte[4];
            var pool = ArrayPool<byte>.Shared;
            while (!cancellationToken.IsCancellationRequested)
            {
                await ReadExactAsync(pipe, header, cancellationToken);
                var len = BinaryPrimitives.ReadInt32LittleEndian(header);
                if (len <= 0)
                {
                    continue;
                }
                if (len > 64 * 1024 * 1024)
                {
                    throw new InvalidOperationException($"Engine frame too large: {len} bytes");
                }

                var data = pool.Rent(len);
                try
                {
                    await ReadExactAsync(pipe, data.AsMemory(0, len), cancellationToken);

                    JsonObject? obj;
                    try
                    {
                        obj = JsonSerializer.Deserialize<JsonObject>(data.AsSpan(0, len), JsonOptions);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, "Failed to parse engine pipe frame ({Len} bytes)", len);
                        continue;
                    }

                    var id = obj?["id"]?.GetValue<string>();
                    if (id is null)
                    {
                        continue;
                    }

                    var hasOk = obj?["ok"] is not null;
                    if (hasOk && _pending.TryGetValue(id, out var tcs))
                    {
                        tcs.TrySetResult(obj!);
                        if (_streams.TryGetValue(id, out var writer))
                        {
                            writer.TryComplete();
                        }
                        continue;
                    }

                    var eventName = obj?["event"]?.GetValue<string>();
                    if (!string.IsNullOrWhiteSpace(eventName) && _streams.TryGetValue(id, out var streamWriter))
                    {
                        streamWriter.TryWrite(obj!.DeepClone() as JsonObject ?? obj!);
                    }
                }
                finally
                {
                    pool.Return(data);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // normal shutdown
        }
        catch (EndOfStreamException)
        {
            // Pipe closed during shutdown or engine restart.
            if (cancellationToken.IsCancellationRequested || _process is null || _process.HasExited)
            {
                return;
            }
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Engine pipe pump failed.");
            FailAllPending(ex);
        }
    }

    private static string CreatePipeName(string prefix)
    {
        const int maxPathLen = 104; // UnixDomainSocketEndPoint limit on macOS
        const string socketPrefix = "CoreFxPipe_";
        var temp = Path.GetTempPath();
        var maxNameLen = maxPathLen - temp.Length - socketPrefix.Length;
        maxNameLen = Math.Clamp(maxNameLen, 12, 64);

        var guid = Guid.NewGuid().ToString("N");
        prefix = string.IsNullOrWhiteSpace(prefix) ? "srdb" : prefix.Trim();
        if (prefix.Length > 16)
        {
            prefix = prefix[..16];
        }
        var suffixLen = Math.Max(4, maxNameLen - (prefix.Length + 1));
        suffixLen = Math.Min(suffixLen, guid.Length);
        return $"{prefix}_{guid[..suffixLen]}";
    }

    private bool IsPipeTransport()
    {
        var t = _transport.Trim().ToLowerInvariant();
        return t is "pipe" or "pipes" or "namedpipe" or "namedpipes";
    }

    private async Task EnsureStartedAsync(CancellationToken cancellationToken)
    {
        if (_process is { HasExited: false })
        {
            if (!IsPipeTransport())
            {
                return;
            }

            if (_pipe is { IsConnected: true } && _blobPipe is { IsConnected: true })
            {
                return;
            }
        }

        await _startLock.WaitAsync(cancellationToken);
        try
        {
            if (_process is { HasExited: false })
            {
                // Process is alive but pipe may not be connected yet.
            }
            else
            {
                StartProcess();
            }

            if (!IsPipeTransport())
            {
                return;
            }

            var pipe = _pipe;
            if (pipe is null)
            {
                throw new InvalidOperationException("Pipe transport selected but no pipe was created.");
            }

            if (!pipe.IsConnected)
            {
                using var connectCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                connectCts.CancelAfter(TimeSpan.FromSeconds(5));
                await pipe.WaitForConnectionAsync(connectCts.Token);
                _logger.LogInformation("Python engine connected via named pipe ({PipeName})", _pipeName);
            }

            var blobPipe = _blobPipe;
            if (blobPipe is null)
            {
                throw new InvalidOperationException("Pipe transport selected but no blob pipe was created.");
            }
            if (!blobPipe.IsConnected)
            {
                using var connectCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                connectCts.CancelAfter(TimeSpan.FromSeconds(5));
                await blobPipe.WaitForConnectionAsync(connectCts.Token);
                _logger.LogInformation("Python engine connected via blob pipe ({PipeName})", _blobPipeName);
            }

            if (_pipePump is null)
            {
                _pipePump = Task.Run(() => PumpPipeAsync(pipe, _cts.Token));
            }
        }
        finally
        {
            _startLock.Release();
        }
    }

    private void EnsureStarted()
    {
        // Obsolete: prefer EnsureStartedAsync. Kept for legacy call sites.
        EnsureStartedAsync(CancellationToken.None).GetAwaiter().GetResult();
    }

    private void StartProcess()
    {
        FailAllPending(new InvalidOperationException("Python engine restarted."));

        // Tear down any previous pipe transport.
        try
        {
            _pipe?.Dispose();
        }
        catch
        {
            // ignore
        }
        try
        {
            _blobPipe?.Dispose();
        }
        catch
        {
            // ignore
        }
        _pipe = null;
        _pipeName = null;
        _pipePump = null;
        _blobPipe = null;
        _blobPipeName = null;

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

        if (IsPipeTransport())
        {
            _pipeName = CreatePipeName("srdb");
            _blobPipeName = CreatePipeName("srdbb");
            _pipe = new NamedPipeServerStream(
                _pipeName,
                PipeDirection.InOut,
                maxNumberOfServerInstances: 1,
                PipeTransmissionMode.Byte,
                PipeOptions.Asynchronous);
            _blobPipe = new NamedPipeServerStream(
                _blobPipeName,
                PipeDirection.InOut,
                maxNumberOfServerInstances: 1,
                PipeTransmissionMode.Byte,
                PipeOptions.Asynchronous);
            startInfo.EnvironmentVariables["DUALBRAIN_PIPE_NAME"] = _pipeName;
            startInfo.EnvironmentVariables["DUALBRAIN_PIPE_CONNECT_TIMEOUT_MS"] = "5000";
            startInfo.EnvironmentVariables["DUALBRAIN_BLOB_PIPE_NAME"] = _blobPipeName;
            startInfo.EnvironmentVariables["DUALBRAIN_BLOB_PIPE_CONNECT_TIMEOUT_MS"] = "5000";
        }

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

                if (IsPipeTransport())
                {
                    _logger.LogDebug("engine stdout: {Line}", line);
                    continue;
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

                var hasOk = obj?["ok"] is not null;
                if (hasOk && _pending.TryGetValue(id, out var tcs))
                {
                    tcs.TrySetResult(obj!);
                    if (_streams.TryGetValue(id, out var writer))
                    {
                        writer.TryComplete();
                    }
                    continue;
                }

                var eventName = obj?["event"]?.GetValue<string>();
                if (!string.IsNullOrWhiteSpace(eventName) && _streams.TryGetValue(id, out var streamWriter))
                {
                    streamWriter.TryWrite(obj!.DeepClone() as JsonObject ?? obj!);
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

        foreach (var item in _streams)
        {
            try
            {
                item.Value.TryComplete(ex);
            }
            catch
            {
                // ignore
            }
        }
        _streams.Clear();
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

        try
        {
            if (_pipePump is not null)
            {
                await _pipePump;
            }
        }
        catch
        {
            // ignore
        }

        try
        {
            _pipe?.Dispose();
        }
        catch
        {
            // ignore
        }
        try
        {
            _blobPipe?.Dispose();
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
        _startLock.Dispose();
        _blobLock.Dispose();
    }
}
