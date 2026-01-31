using System.Text.Json.Nodes;
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

builder.Services.AddHostedService<EngineWarmupService>();

var app = builder.Build();

app.UseResponseCompression();
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/favicon.ico", () => Results.NoContent());

app.MapGet("/v1/health", async (PythonEngineClient engine, CancellationToken ct) =>
{
    var result = await engine.CallAsync("health", new JsonObject(), ct);
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

    var result = await engine.CallAsync("process", body, ct);

    return Results.Json(result);
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
