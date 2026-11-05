# push.ps1 —— 把工作区推到 GitHub
#
# 用法（在任意目录）：
#   powershell -ExecutionPolicy Bypass -File scripts\push.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\push.ps1 -Owner RUIKER -RepoName embodied-ai-workspace
#
# 前提：仓库已在网页上建好（https://github.com/new），不要勾 README / .gitignore / license。
# 第一次推送时 GCM 会弹出浏览器登录，之后会记住凭据。

param(
    [string]$Owner = "RUIKER",
    [string]$RepoName = "embodied-ai-workspace",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "找不到 git，请先安装 Git for Windows。" -ForegroundColor Red
    exit 1
}

$url = "https://github.com/$Owner/$RepoName.git"

# 确保 origin 指向目标仓库
$current = (git remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0) {
    git remote add origin $url
    Write-Host "已添加 origin -> $url"
} elseif ($current -ne $url) {
    git remote set-url origin $url
    Write-Host "已更新 origin -> $url"
}

Write-Host "目标仓库：$url"
Write-Host "当前分支：$Branch"

# 有未提交的改动时先问一句
$status = git status --porcelain
if ($status) {
    Write-Host "工作区有未提交的改动：" -ForegroundColor Yellow
    $status | Select-Object -First 20
    $answer = Read-Host "是否先提交这些改动？(y/N)"
    if ($answer -eq "y") {
        $msg = Read-Host "提交信息（留空用默认）"
        if (-not $msg) { $msg = "chore: 同步工作区" }
        git add -A
        git commit -m "$msg`n`nCo-Authored-By: Claude Code <noreply@anthropic.com>"
    }
}

Write-Host "开始推送……第一次会弹出浏览器登录 GitHub。"
git push -u origin $Branch

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "推送失败。常见原因：" -ForegroundColor Yellow
    Write-Host "  1. 仓库还没建：打开 https://github.com/new 新建 $RepoName（不要勾任何初始化选项）"
    Write-Host "  2. 登录凭据失效：执行 git config --global credential.helper manager 后重试"
    Write-Host "  3. 代理干扰：Clash / FlClash 全局模式会让 GitHub TLS 握手失败，改用规则模式或直连"
    exit 1
}

Write-Host "完成：https://github.com/$Owner/$RepoName/tree/$Branch" -ForegroundColor Green
