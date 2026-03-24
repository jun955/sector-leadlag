# ==========================================================
#  sector-leadlag タスクスケジューラ登録スクリプト
#  使い方: PowerShell を管理者として開き、
#          .\register_task.ps1 を実行する
# ==========================================================

$TaskName    = "SectorLeadLag_DailyUpdate"
$BatFile     = "C:\trade-tools\sector-leadlag\run_update.bat"
$WorkDir     = "C:\trade-tools\sector-leadlag"
$RunTime     = "06:30"

# --- アクション: cmd.exe 経由でバッチファイルを実行 ---
# (バッチファイルを直接 Execute に指定するとウィンドウが閉じずログが残らない場合があるため cmd /c を使用)
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`" >> `"$WorkDir\logs\scheduler.log`" 2>&1" `
    -WorkingDirectory $WorkDir

# --- トリガー: 月〜金 06:30 ---
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $RunTime

# --- 設定 ---
$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `                          # スリープ中でも起こして実行
    -StartWhenAvailable `                 # 起動遅延時に次回起動時に実行
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `   # 最大1時間で強制終了
    -MultipleInstances IgnoreNew `        # 二重起動防止
    -Priority 7                           # 通常優先度

# --- 登録（既存タスクは上書き） ---
Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $Action `
    -Trigger   $Trigger `
    -Settings  $Settings `
    -RunLevel  Highest `
    -Force

Write-Host ""
Write-Host "=== 登録完了 ===" -ForegroundColor Green
Write-Host "タスク名  : $TaskName"
Write-Host "実行時刻  : 毎週月〜金 $RunTime"
Write-Host "実行ファイル: $BatFile"
Write-Host "ログ保存先 : $WorkDir\logs\"
Write-Host ""
Write-Host "確認コマンド:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host ""
Write-Host "手動テスト実行:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "削除する場合:"
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
