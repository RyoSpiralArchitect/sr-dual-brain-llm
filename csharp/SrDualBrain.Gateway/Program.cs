using System.Text.Json.Nodes;
using System.Text.Json;
using Microsoft.AspNetCore.ResponseCompression;
using SrDualBrain.Gateway;

static string? FindRepoRoot(string startDir)
{
    var dir = new DirectoryInfo(startDir);
    for (var i = 0; i < 10 && dir is not null; i++, dir = dir.Parent)
    {
        var candidate = Path.Combine(dir.FullName, "sr-dual-brain-llm", "scripts", "engine_stdio.py");
        if (File.Exists(candidate))
        {
            return dir.FullName;
        }
    }
    return null;
}

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddResponseCompression(options =>
{
    options.MimeTypes = ResponseCompressionDefaults.MimeTypes.Concat(new[] { "application/json" });
});

var repoRoot = Environment.GetEnvironmentVariable("DUALBRAIN_REPO_ROOT")
              ?? FindRepoRoot(Directory.GetCurrentDirectory())
              ?? FindRepoRoot(AppContext.BaseDirectory)
              ?? Directory.GetCurrentDirectory();
var pythonExe = Environment.GetEnvironmentVariable("DUALBRAIN_PYTHON") ?? "python3";
var engineScript = Environment.GetEnvironmentVariable("DUALBRAIN_ENGINE_PATH")
                   ?? Path.Combine(repoRoot, "sr-dual-brain-llm", "scripts", "engine_stdio.py");

if (!File.Exists(engineScript))
{
    throw new FileNotFoundException(
        $"Python engine script not found: {engineScript}\n" +
        "Set DUALBRAIN_ENGINE_PATH to the full path of engine_stdio.py or set DUALBRAIN_REPO_ROOT to the repo root.");
}

builder.Services.AddSingleton<PythonEngineClient>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<PythonEngineClient>>();
    var timeoutRaw = Environment.GetEnvironmentVariable("DUALBRAIN_ENGINE_TIMEOUT_SECONDS");
    var timeoutSeconds = 120;
    if (!string.IsNullOrWhiteSpace(timeoutRaw) && int.TryParse(timeoutRaw, out var parsed))
    {
        timeoutSeconds = Math.Clamp(parsed, 5, 600);
    }
    return new PythonEngineClient(
        logger,
        pythonExe: pythonExe,
        engineScriptPath: engineScript,
        workingDirectory: repoRoot,
        defaultTimeout: TimeSpan.FromSeconds(timeoutSeconds));
});

builder.Services.AddSingleton<EngineHealthCache>();
builder.Services.AddHostedService<EngineWarmupService>();

var app = builder.Build();

app.UseResponseCompression();
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/favicon.ico", () => Results.NoContent());

app.MapGet("/v1/health", async (EngineHealthCache engineHealth, CancellationToken ct) =>
{
    var result = await engineHealth.GetAsync(ct);
    return Results.Json(new JsonObject
    {
        ["gateway"] = "ok",
        ["engine"] = result,
    });
});

app.MapPost("/v1/reset", async (PythonEngineClient engine, JsonObject body, CancellationToken ct) =>
{
    var sessionId = body["session_id"]?.GetValue<string>() ?? "default";
    var result = await engine.CallAsync("reset", new JsonObject { ["session_id"] = sessionId }, ct);
    return Results.Json(result);
});

app.MapPost("/v1/process", async (PythonEngineClient engine, JsonObject body, CancellationToken ct) =>
{
    body["session_id"] ??= "default";
    body["leading_brain"] ??= "auto";
    body["return_dialogue_flow"] ??= true;
    body["qid"] ??= Guid.NewGuid().ToString();

    var result = await engine.CallAsync("process", body, ct);

    return Results.Json(result);
});

app.MapGet("/v1/trace/{qid}", async (PythonEngineClient engine, HttpRequest request, string qid, CancellationToken ct) =>
{
    var sessionId = request.Query["session_id"].ToString();
    if (string.IsNullOrWhiteSpace(sessionId))
    {
        sessionId = "default";
    }

    var includeTelemetry = ParseBool(request.Query["include_telemetry"], defaultValue: true);
    var includeDialogueFlow = ParseBool(request.Query["include_dialogue_flow"], defaultValue: true);
    var includeExecutive = ParseBool(request.Query["include_executive"], defaultValue: true);

    var result = await engine.CallAsync(
        "get_trace",
        new JsonObject
        {
            ["session_id"] = sessionId,
            ["qid"] = qid,
            ["include_telemetry"] = includeTelemetry,
            ["include_dialogue_flow"] = includeDialogueFlow,
            ["include_executive"] = includeExecutive,
        },
        ct);

    var found = result["found"]?.GetValue<bool>() ?? false;
    return found ? Results.Json(result) : Results.NotFound(result);
});

app.MapPost("/v1/process/stream", async (HttpContext ctx, PythonEngineClient engine, JsonObject body, CancellationToken ct) =>
{
    ctx.Response.Headers.CacheControl = "no-cache";
    ctx.Response.Headers.Pragma = "no-cache";
    ctx.Response.ContentType = "text/event-stream; charset=utf-8";

    body["session_id"] ??= "default";
    body["leading_brain"] ??= "auto";
    body["qid"] ??= Guid.NewGuid().ToString();
    body["return_telemetry"] = false;
    body["return_dialogue_flow"] = false;

    var qid = body["qid"]?.GetValue<string>() ?? "";
    var sessionId = body["session_id"]?.GetValue<string>() ?? "default";

    await WriteSseAsync(ctx.Response, "start", new JsonObject { ["qid"] = qid, ["session_id"] = sessionId }, ct);

    JsonObject? finalEnvelope = null;
    try
    {
        await foreach (var msg in engine.CallStreamAsync("process_stream", body, ct))
        {
            if (msg["ok"] is not null)
            {
                finalEnvelope = msg;
                break;
            }

            var evName = msg["event"]?.GetValue<string>() ?? "";
            if (evName == "delta")
            {
                var text = msg["text"]?.GetValue<string>() ?? "";
                if (!string.IsNullOrEmpty(text))
                {
                    await WriteSseAsync(ctx.Response, "delta", new JsonObject { ["text"] = text }, ct);
                }
            }
            else if (evName == "reset")
            {
                await WriteSseAsync(ctx.Response, "reset", new JsonObject(), ct);
            }
        }
    }
    catch (OperationCanceledException)
    {
        return;
    }
    catch (Exception ex)
    {
        await WriteSseAsync(ctx.Response, "error", new JsonObject { ["message"] = ex.Message }, ct);
        return;
    }

    if (finalEnvelope is null)
    {
        await WriteSseAsync(ctx.Response, "error", new JsonObject { ["message"] = "No final response from engine." }, ct);
        return;
    }

    var ok = finalEnvelope["ok"]?.GetValue<bool>() ?? false;
    if (!ok)
    {
        var err = finalEnvelope["error"] as JsonObject;
        var errType = err?["type"]?.GetValue<string>() ?? "EngineError";
        var errMsg = err?["message"]?.GetValue<string>() ?? "unknown error";
        await WriteSseAsync(ctx.Response, "error", new JsonObject { ["message"] = $"{errType}: {errMsg}" }, ct);
        return;
    }

    var result = finalEnvelope["result"] as JsonObject;
    var answer = result?["answer"]?.GetValue<string>() ?? "";

    var finalPayload = new JsonObject
    {
        ["qid"] = result?["qid"]?.GetValue<string>() ?? qid,
        ["answer"] = answer,
        ["session_id"] = result?["session_id"]?.GetValue<string>() ?? sessionId,
        ["metrics"] = result?["metrics"]?.DeepClone(),
    };
    await WriteSseAsync(ctx.Response, "final", finalPayload, ct);
    await WriteSseAsync(ctx.Response, "done", new JsonObject(), ct);
});

app.MapPost("/v1/episodes/search", async (PythonEngineClient engine, JsonObject body, CancellationToken ct) =>
{
    body["session_id"] ??= "default";
    body["topk"] ??= 5;
    body["candidate_limit"] ??= 500;

    var result = await engine.CallAsync("search_episodes", body, ct);
    return Results.Json(result);
});

app.MapPost("/v1/telemetry/query", async (PythonEngineClient engine, JsonObject body, CancellationToken ct) =>
{
    body["session_id"] ??= "default";
    body["limit"] ??= 250;

    var result = await engine.CallAsync("query_telemetry", body, ct);
    return Results.Json(result);
});

app.MapPost("/v1/schema/list", async (PythonEngineClient engine, JsonObject body, CancellationToken ct) =>
{
    body["session_id"] ??= "default";
    body["limit"] ??= 16;

    var result = await engine.CallAsync("list_schema_memories", body, ct);
    return Results.Json(result);
});

app.Run();

static bool ParseBool(string? raw, bool defaultValue)
{
    if (string.IsNullOrWhiteSpace(raw))
    {
        return defaultValue;
    }
    return raw.Trim().ToLowerInvariant() switch
    {
        "1" or "true" or "yes" or "on" => true,
        "0" or "false" or "no" or "off" => false,
        _ => defaultValue,
    };
}

static async Task WriteSseAsync(HttpResponse response, string @event, JsonObject payload, CancellationToken ct)
{
    var data = JsonSerializer.Serialize(payload, new JsonSerializerOptions(JsonSerializerDefaults.Web) { WriteIndented = false });
    await response.WriteAsync($"event: {@event}\n", ct);
    await response.WriteAsync($"data: {data}\n\n", ct);
    await response.Body.FlushAsync(ct);
}
