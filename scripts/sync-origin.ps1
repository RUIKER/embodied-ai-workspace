# sync-origin.ps1 —— 把工作区里的某个子项目同步回它原来的 GitHub 仓库
#
# 背景：工作区仓库只有一个初始提交，和两个子仓库没有共同历史，所以
#       git subtree push 用不了（会被当成无关历史拒绝）。这个脚本改成
#       "从备份的 .git 克隆一份 → 用工作区文件覆盖 → 提交 → 推送"，
#       这样原子仓库的提交历史保持不变。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\sync-origin.ps1              # 预览差异（不推送）
#   powershell -ExecutionPolicy Bypass -File scripts\sync-origin.ps1 -Push        # 提交并推送 rl 仓库
#   powershell -ExecutionPolicy Bypass -File scripts\sync-origin.ps1 -Repo search -Push
#
# 预览模式会在临时目录里生成一份待推送的副本，列出差异后停下，方便先检查。

param(
    [ValidateSet("rl", "search")]
    [string]$Repo = "rl",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$workspace = Split-Path -Parent $PSScriptRoot
$backupRoot = Join-Path $env:USERPROFILE "Desktop\_embodied_git_backup"

$config = @{
    rl = @{
        Folder = "hands-on-reinforcement-learning"
        Backup = Join-Path $backupRoot "hands-on-reinforcement-learning.git"
        Url    = "https://github.com/RUIKER/hands-on-reinforcement-learning.git"
    }
    search = @{
        Folder = "industrial-search-agent"
        Backup = Join-Path $backupRoot "industrial-search-agent.git"
        Url    = "https://github.com/RUIKER/industrial-search-agent.git"
    }
}

$item = $config[$Repo]
$src = Join-Path $workspace $item.Folder

if (-not (Test-Path $src)) { throw "找不到源目录：$src" }
if (-not (Test-Path $item.Backup)) {
    throw "找不到备份的 .git：$($item.Backup)。没有它无法保留原仓库历史，请先恢复该目录。"
}

$tmp = Join-Path $env:TEMP ("sync-" + $Repo + "-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
Write-Host "克隆原仓库到临时目录：$tmp"
git clone --quiet --no-hardlinks $item.Backup $tmp
if ($LASTEXITCODE -ne 0) { throw "克隆失败" }
git -C $tmp remote set-url origin $item.Url

# 原仓库在 .git/info/exclude 里有一些只在本机生效的忽略规则（如 output/），
# clone 不会带走它们，这里补上，避免把本机产物同步上去
$localExclude = Join-Path $item.Backup "info\exclude"
if (Test-Path $localExclude) {
    Copy-Item $localExclude (Join-Path $tmp ".git\info\exclude") -Force
}

# 清空工作区文件，只保留 .git
Get-ChildItem -LiteralPath $tmp -Force |
    Where-Object { $_.Name -ne ".git" } |
    Remove-Item -Recurse -Force

# 用工作区文件覆盖过去；.gitignore 仍会拦住 .venv 和大文件
& robocopy $src $tmp /E /NFL /NDL /NJH /NJS /XD .git .venv __pycache__ .pytest_cache .ipynb_checkpoints logs /XF *.pyc *.pt *.log | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy 失败，退出码 $LASTEXITCODE" }

git -c core.safecrlf=false -C $tmp add -A
$changes = git -C $tmp status --porcelain
if (-not $changes) {
    Write-Host "没有差异，原仓库已经是最新。" -ForegroundColor Green
    Remove-Item -Recurse -Force $tmp
    exit 0
}

Write-Host ""
Write-Host "待同步的改动：" -ForegroundColor Yellow
$changes | Select-Object -First 60
$count = ($changes | Measure-Object).Count
Write-Host "（共 $count 项）"

if ($Repo -eq "search") {
    Write-Host ""
    Write-Host "提醒：industrial-search-agent 的历史里仍有 docs/materials 下的语料原件，" -ForegroundColor Yellow
    Write-Host "      本次同步不会改变历史。要清除需要单独用 git filter-repo 处理。" -ForegroundColor Yellow
}

if (-not $Push) {
    Write-Host ""
    Write-Host "预览结束，未提交、未推送。确认无误后加 -Push 重跑。" -ForegroundColor Cyan
    Write-Host "临时副本保留在：$tmp"
    exit 0
}

git -C $tmp commit -m "sync: 从工作区同步 $(Get-Date -Format 'yyyy-MM-dd HH:mm')`n`nCo-Authored-By: Claude Code <noreply@anthropic.com>"
if ($LASTEXITCODE -ne 0) { throw "提交失败" }

git -C $tmp push origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "推送失败，临时副本保留在：$tmp" -ForegroundColor Red
    exit 1
}

Remove-Item -Recurse -Force $tmp
Write-Host "已同步到 $($item.Url)" -ForegroundColor Green
