using Avalonia;
using Avalonia.Controls;
using Avalonia.Input.Platform;
using Avalonia.Layout;
using Avalonia.Media;

namespace ProtectorOperator;

public sealed class MainWindow : Window
{
    private readonly PhBackend _backend;
    private readonly ComboBox _repoSelector = new() { MinWidth = 420 };
    private readonly TextBox _taskEditor = CreateEditor("Paste or write the task here.");
    private readonly TextBox _executionPlanPanel = CreateReadOnlyPanel("Selected profile, agents, skills, and safety gates appear here.");
    private readonly TextBox _promptPanel = CreateReadOnlyPanel("Generated Codex prompt appears here.");
    private readonly TextBox _codexOutputEditor = CreateEditor("Paste Codex output here.");
    private readonly TextBox _reviewPanel = CreateReadOnlyPanel("Review findings or reviewer prompt appears here.");
    private readonly TextBlock _status = new() { Text = "Loading repo aliases...", VerticalAlignment = VerticalAlignment.Center };
    private readonly Button _planButton = new() { Content = "Generate Execution Plan" };
    private readonly Button _generateButton = new() { Content = "Generate Codex Prompt" };
    private readonly Button _reviewButton = new() { Content = "Deterministic Review" };
    private readonly Button _reviewerPromptButton = new() { Content = "Generate Codex Reviewer Prompt" };
    private string _taskMode = "(none)";
    private string _executionProfile = "(none)";
    private string _agentSummary = "(none)";
    private string _skillSummary = "(none)";
    private string _gateSummary = "(none)";
    private int _selectedContextCount;
    private string _clipboardStatus = "Clipboard: idle";

    public MainWindow(PhBackend backend)
    {
        _backend = backend;
        Title = "Protector Operator";
        Width = 1280;
        Height = 860;
        MinWidth = 980;
        MinHeight = 700;
        Content = BuildLayout();

        _planButton.Click += async (_, _) => await GenerateExecutionPlanAsync();
        _generateButton.Click += async (_, _) => await GeneratePromptAsync();
        _reviewButton.Click += async (_, _) => await ReviewAsync();
        _reviewerPromptButton.Click += async (_, _) => await GenerateReviewerPromptAsync();
        Opened += async (_, _) => await LoadReposAsync();
    }

    private Control BuildLayout()
    {
        Grid root = new()
        {
            Margin = new Thickness(12),
            RowDefinitions =
            {
                new RowDefinition(GridLength.Auto),
                new RowDefinition(new GridLength(1, GridUnitType.Star)),
                new RowDefinition(GridLength.Auto),
            },
        };

        StackPanel toolbar = new()
        {
            Orientation = Orientation.Horizontal,
            Spacing = 10,
            VerticalAlignment = VerticalAlignment.Center,
        };
        toolbar.Children.Add(new TextBlock { Text = "Repo", VerticalAlignment = VerticalAlignment.Center });
        toolbar.Children.Add(_repoSelector);
        toolbar.Children.Add(_planButton);
        toolbar.Children.Add(_generateButton);
        toolbar.Children.Add(_reviewButton);
        toolbar.Children.Add(_reviewerPromptButton);

        Grid workspace = new()
        {
            RowDefinitions =
            {
                new RowDefinition(new GridLength(1, GridUnitType.Star)),
                new RowDefinition(new GridLength(1, GridUnitType.Star)),
                new RowDefinition(new GridLength(1, GridUnitType.Star)),
            },
            ColumnDefinitions =
            {
                new ColumnDefinition(new GridLength(1, GridUnitType.Star)),
                new ColumnDefinition(new GridLength(1, GridUnitType.Star)),
            },
            Margin = new Thickness(0, 12, 0, 12),
        };

        AddPanel(workspace, "Task", _taskEditor, row: 0, column: 0);
        AddPanel(workspace, "Generated Prompt", _promptPanel, row: 0, column: 1);
        AddPanel(workspace, "Execution Plan", _executionPlanPanel, row: 1, column: 0, columnSpan: 2);
        AddPanel(workspace, "Codex Output", _codexOutputEditor, row: 2, column: 0);
        AddPanel(workspace, "Review / Reviewer Prompt", _reviewPanel, row: 2, column: 1);

        Border statusBar = new()
        {
            BorderBrush = Brushes.LightGray,
            BorderThickness = new Thickness(1, 0, 0, 0),
            Padding = new Thickness(0, 8, 0, 0),
            Child = _status,
        };

        Grid.SetRow(toolbar, 0);
        Grid.SetRow(workspace, 1);
        Grid.SetRow(statusBar, 2);
        root.Children.Add(toolbar);
        root.Children.Add(workspace);
        root.Children.Add(statusBar);
        return root;
    }

    private static void AddPanel(Grid parent, string title, TextBox editor, int row, int column, int columnSpan = 1)
    {
        Grid panel = new()
        {
            Margin = new Thickness(6),
            RowDefinitions =
            {
                new RowDefinition(GridLength.Auto),
                new RowDefinition(new GridLength(1, GridUnitType.Star)),
            },
        };
        panel.Children.Add(new TextBlock
        {
            Text = title,
            FontWeight = FontWeight.SemiBold,
            Margin = new Thickness(0, 0, 0, 6),
        });
        Grid.SetRow(editor, 1);
        panel.Children.Add(editor);
        Grid.SetRow(panel, row);
        Grid.SetColumn(panel, column);
        if (columnSpan > 1)
        {
            Grid.SetColumnSpan(panel, columnSpan);
        }
        parent.Children.Add(panel);
    }

    private async Task LoadReposAsync()
    {
        try
        {
            IReadOnlyList<RepoAlias> repos = await _backend.LoadReposAsync(CancellationToken.None);
            _repoSelector.ItemsSource = repos;
            _repoSelector.SelectedItem = repos.FirstOrDefault(repo => repo.Name == "FinanciacionCore") ?? repos.FirstOrDefault();
            SetStatus("Repo aliases loaded.");
        }
        catch (Exception exc)
        {
            SetStatus($"Failed to load repos: {exc.Message}");
        }
    }

    private async Task GeneratePromptAsync()
    {
        await RunUiActionAsync(async () =>
        {
            RepoAlias repo = RequireRepo();
            string task = RequireText(_taskEditor, "Task is required.");
            ExecutionPlanResult plan = await _backend.GenerateExecutionPlanAsync(repo.Name, task, CancellationToken.None);
            ApplyExecutionPlan(plan);
            PromptResult result = await _backend.GeneratePromptAsync(repo.Name, task, CancellationToken.None);
            _promptPanel.Text = result.Prompt;
            _taskMode = result.TaskMode;
            _selectedContextCount = result.SelectedContextCount;
            _clipboardStatus = await CopyToClipboardAsync(result.Prompt, "Codex prompt copied to clipboard.");
            SetStatus("Generated Codex prompt.");
        });
    }

    private async Task GenerateExecutionPlanAsync()
    {
        await RunUiActionAsync(async () =>
        {
            RepoAlias repo = RequireRepo();
            string task = RequireText(_taskEditor, "Task is required.");
            ExecutionPlanResult plan = await _backend.GenerateExecutionPlanAsync(repo.Name, task, CancellationToken.None);
            ApplyExecutionPlan(plan);
            SetStatus("Generated execution plan. Codex execution remains disabled.");
        });
    }

    private async Task ReviewAsync()
    {
        await RunUiActionAsync(async () =>
        {
            RepoAlias repo = RequireRepo();
            string task = RequireText(_taskEditor, "Task is required.");
            string output = RequireText(_codexOutputEditor, "Codex output is required.");
            ReviewResult result = await _backend.ReviewAsync(repo.Name, task, output, CancellationToken.None);
            _reviewPanel.Text = result.Text;
            SetStatus($"Review status: {result.Status}");
        });
    }

    private async Task GenerateReviewerPromptAsync()
    {
        await RunUiActionAsync(async () =>
        {
            RepoAlias repo = RequireRepo();
            string task = RequireText(_taskEditor, "Task is required.");
            string output = RequireText(_codexOutputEditor, "Codex output is required.");
            PromptResult result = await _backend.GenerateReviewerPromptAsync(repo.Name, task, output, CancellationToken.None);
            _reviewPanel.Text = result.Prompt;
            _clipboardStatus = await CopyToClipboardAsync(result.Prompt, "Codex reviewer prompt copied to clipboard.");
            SetStatus("Generated Codex reviewer prompt.");
        });
    }

    private void ApplyExecutionPlan(ExecutionPlanResult plan)
    {
        _executionPlanPanel.Text = plan.Text;
        _executionProfile = plan.Profile;
        _agentSummary = string.Join(", ", plan.Agents);
        _skillSummary = string.Join(", ", plan.Skills);
        _gateSummary = $"{plan.SafetyGates.Count} safety gates";
    }

    private async Task RunUiActionAsync(Func<Task> action)
    {
        SetBusy(isBusy: true);
        try
        {
            await action();
        }
        catch (Exception exc)
        {
            SetStatus(exc.Message);
        }
        finally
        {
            SetBusy(isBusy: false);
        }
    }

    private void SetBusy(bool isBusy)
    {
        _planButton.IsEnabled = !isBusy;
        _generateButton.IsEnabled = !isBusy;
        _reviewButton.IsEnabled = !isBusy;
        _reviewerPromptButton.IsEnabled = !isBusy;
        _repoSelector.IsEnabled = !isBusy;
    }

    private RepoAlias RequireRepo()
    {
        if (_repoSelector.SelectedItem is RepoAlias repo)
        {
            return repo;
        }

        throw new InvalidOperationException("Select a repo first.");
    }

    private static string RequireText(TextBox editor, string message)
    {
        string text = editor.Text?.Trim() ?? "";
        if (text.Length == 0)
        {
            throw new InvalidOperationException(message);
        }

        return text;
    }

    private async Task<string> CopyToClipboardAsync(string text, string success)
    {
        IClipboard? clipboard = GetTopLevel(this)?.Clipboard;
        if (clipboard is null)
        {
            return "Clipboard: unavailable";
        }

        await clipboard.SetTextAsync(text);
        return $"Clipboard: {success}";
    }

    private void SetStatus(string message)
    {
        string repo = _repoSelector.SelectedItem is RepoAlias selected ? selected.Name : "(none)";
        _status.Text =
            $"Repo: {repo} | Profile: {_executionProfile} | Agents: {_agentSummary} | Skills: {_skillSummary} | Gates: {_gateSummary} | Task mode: {_taskMode} | Selected context: {_selectedContextCount} | {_clipboardStatus} | {message}";
    }

    private static TextBox CreateEditor(string watermark)
    {
        return new TextBox
        {
            AcceptsReturn = true,
            TextWrapping = TextWrapping.Wrap,
            Watermark = watermark,
            FontFamily = FontFamily.Parse("Consolas"),
            FontSize = 13,
        };
    }

    private static TextBox CreateReadOnlyPanel(string watermark)
    {
        TextBox textBox = CreateEditor(watermark);
        textBox.IsReadOnly = true;
        return textBox;
    }
}
