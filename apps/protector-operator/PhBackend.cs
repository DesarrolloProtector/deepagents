using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace ProtectorOperator;

public sealed record RepoAlias(string Name, string Path)
{
    public override string ToString()
    {
        return $"{Name} - {Path}";
    }
}

public sealed record PromptResult(string Prompt, string TaskMode, int SelectedContextCount);

public sealed record ReviewResult(string Text, string Status);

public sealed class PhBackend
{
    private static readonly Encoding Utf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
    private readonly string _packageDirectory;

    public PhBackend()
    {
        _packageDirectory = FindPackageDirectory();
    }

    public async Task<IReadOnlyList<RepoAlias>> LoadReposAsync(CancellationToken cancellationToken)
    {
        ProcessResult result = await RunPhAsync(["repos"], cancellationToken);
        if (result.ExitCode != 0)
        {
            throw new InvalidOperationException(DescribeFailure("ph repos", result));
        }

        List<RepoAlias> aliases = [];
        foreach (string line in result.Stdout.SplitLines())
        {
            Match match = Regex.Match(line, @"^\s*-\s*(?<name>[^:]+):\s*(?<path>.+?)\s*$");
            if (match.Success)
            {
                aliases.Add(new RepoAlias(match.Groups["name"].Value, match.Groups["path"].Value));
            }
        }

        return aliases;
    }

    public async Task<PromptResult> GeneratePromptAsync(string repo, string task, CancellationToken cancellationToken)
    {
        string path = CreateTempPath(".md");
        try
        {
            ProcessResult result = await RunPhAsync(
                ["task", repo, "--output", path, "--overwrite", "--no-copy", task],
                cancellationToken
            );
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(DescribeFailure("ph task", result));
            }

            string prompt = await File.ReadAllTextAsync(path, Utf8, cancellationToken);
            return new PromptResult(
                prompt.TrimEnd(),
                ParseTaskMode(prompt),
                ParseSelectedContextCount(prompt)
            );
        }
        finally
        {
            DeleteTempFile(path);
        }
    }

    public async Task<ReviewResult> ReviewAsync(string repo, string task, string codexOutput, CancellationToken cancellationToken)
    {
        string path = await WriteCodexOutputAsync(codexOutput, cancellationToken);
        try
        {
            ProcessResult result = await RunPhAsync(["review", repo, "--codex-output", path, task], cancellationToken);
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(DescribeFailure("ph review", result));
            }

            return new ReviewResult(result.Stdout.TrimEnd(), ParseReviewStatus(result.Stdout));
        }
        finally
        {
            DeleteTempFile(path);
        }
    }

    public async Task<PromptResult> GenerateReviewerPromptAsync(string repo, string task, string codexOutput, CancellationToken cancellationToken)
    {
        string path = await WriteCodexOutputAsync(codexOutput, cancellationToken);
        try
        {
            ProcessResult result = await RunPhAsync(
                ["review-codex", repo, "--codex-output", path, "--no-copy", task],
                cancellationToken
            );
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(DescribeFailure("ph review-codex", result));
            }

            string prompt = result.Stdout.TrimEnd();
            return new PromptResult(prompt, ParseTaskMode(prompt), ParseSelectedContextCount(prompt));
        }
        finally
        {
            DeleteTempFile(path);
        }
    }

    private async Task<ProcessResult> RunPhAsync(IReadOnlyList<string> args, CancellationToken cancellationToken)
    {
        try
        {
            return await RunProcessAsync("ph", args, _packageDirectory, cancellationToken);
        }
        catch (System.ComponentModel.Win32Exception)
        {
            string[] uvArgs = ["run", "ph", .. args];
            return await RunProcessAsync("uv", uvArgs, _packageDirectory, cancellationToken);
        }
    }

    private static async Task<ProcessResult> RunProcessAsync(
        string fileName,
        IReadOnlyList<string> args,
        string workingDirectory,
        CancellationToken cancellationToken
    )
    {
        ProcessStartInfo info = new()
        {
            FileName = fileName,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        info.Environment["PYTHONUTF8"] = "1";
        info.Environment["PYTHONIOENCODING"] = "utf-8";
        foreach (string arg in args)
        {
            info.ArgumentList.Add(arg);
        }

        using Process process = new() { StartInfo = info };
        process.Start();
        Task<string> stdout = process.StandardOutput.ReadToEndAsync(cancellationToken);
        Task<string> stderr = process.StandardError.ReadToEndAsync(cancellationToken);
        await process.WaitForExitAsync(cancellationToken);
        return new ProcessResult(process.ExitCode, await stdout, await stderr);
    }

    private static async Task<string> WriteCodexOutputAsync(string codexOutput, CancellationToken cancellationToken)
    {
        string path = CreateTempPath(".txt");
        await File.WriteAllTextAsync(path, codexOutput, Utf8, cancellationToken);
        return path;
    }

    private static string CreateTempPath(string extension)
    {
        return Path.Combine(Path.GetTempPath(), $"protector-operator-{Guid.NewGuid():N}{extension}");
    }

    private static void DeleteTempFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private static string ParseTaskMode(string text)
    {
        Match match = Regex.Match(text, @"(?im)^Task mode:\s*(?<mode>\S+)\s*$");
        return match.Success ? match.Groups["mode"].Value : "(unknown)";
    }

    private static string ParseReviewStatus(string text)
    {
        Match match = Regex.Match(text, @"(?im)^Status:\s*(?<status>PASS|REVIEW_NEEDED)\s*$");
        return match.Success ? match.Groups["status"].Value : "(unknown)";
    }

    private static int ParseSelectedContextCount(string prompt)
    {
        Match match = Regex.Match(
            prompt,
            @"(?ims)^Selected context paths:\s*(?<paths>.*?)(?:^Scope boundaries:|^Generated implementation prompt:)"
        );
        if (!match.Success)
        {
            return 0;
        }

        return match.Groups["paths"].Value.SplitLines().Count(line => line.TrimStart().StartsWith("- ", StringComparison.Ordinal));
    }

    private static string DescribeFailure(string command, ProcessResult result)
    {
        string error = string.IsNullOrWhiteSpace(result.Stderr) ? result.Stdout : result.Stderr;
        return $"{command} failed with exit code {result.ExitCode}: {error.Trim()}";
    }

    private static string FindPackageDirectory()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            string candidate = Path.Combine(current.FullName, "libs", "deepagents");
            if (File.Exists(Path.Combine(candidate, "pyproject.toml")))
            {
                return candidate;
            }

            current = current.Parent;
        }

        string cwdCandidate = Path.Combine(Environment.CurrentDirectory, "libs", "deepagents");
        return File.Exists(Path.Combine(cwdCandidate, "pyproject.toml")) ? cwdCandidate : Environment.CurrentDirectory;
    }
}

internal sealed record ProcessResult(int ExitCode, string Stdout, string Stderr);

internal static class StringExtensions
{
    public static string[] SplitLines(this string value)
    {
        return value.Replace("\r\n", "\n", StringComparison.Ordinal).Replace('\r', '\n').Split('\n');
    }
}
