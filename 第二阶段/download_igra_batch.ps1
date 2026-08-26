param(
    [string]$IndexHtml = "",
    [string]$Assignments = "",
    [string]$OutputDir = "",
    [string]$Proxy = "http://127.0.0.1:7897",
    [int]$Limit = 10,
    [int]$MaxTime = 3600
)

$ErrorActionPreference = 'Stop'
$base = 'https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/derived-por'
if([string]::IsNullOrWhiteSpace($IndexHtml)){
    $IndexHtml = (Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File -Filter 'derived_por_index.html' | Select-Object -First 1).FullName
}
if([string]::IsNullOrWhiteSpace($IndexHtml) -or -not (Test-Path -LiteralPath $IndexHtml)){ throw "derived_por_index.html not found" }
if([string]::IsNullOrWhiteSpace($Assignments)){ $Assignments = Join-Path $PSScriptRoot 'result_strict_xlsx_split_20260819\file_assignments.csv' }
if([string]::IsNullOrWhiteSpace($OutputDir)){ $OutputDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'igra_batch_20260824' }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$html = Get-Content -LiteralPath $IndexHtml -Raw -Encoding UTF8
$rows = [regex]::Matches($html, '(?s)<tr>.*?href="([A-Z0-9]{11})-drvd\.txt\.zip".*?<td align="right">(\d+)</td>.*?</tr>') | ForEach-Object {
    [pscustomobject]@{ Station = $_.Groups[1].Value; Bytes = [int64]$_.Groups[2].Value }
}
$requested = Import-Csv -LiteralPath $Assignments -Encoding UTF8 |
    Where-Object { [int]$_.year -ge 2014 -and [int]$_.year -le 2018 } |
    Select-Object -ExpandProperty station_id -Unique
$targets = @($rows | Where-Object {
    ($requested -contains $_.Station) -and
    (-not (Test-Path (Join-Path $OutputDir "$($_.Station)-drvd.txt.zip")) -or
     (Get-Item (Join-Path $OutputDir "$($_.Station)-drvd.txt.zip")).Length -ne [int64]$_.Bytes)
} | Sort-Object Bytes | Select-Object -First $Limit)
$manifest = Join-Path $OutputDir 'download_manifest.csv'
$targets | Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8
$log = Join-Path $OutputDir 'download_log.csv'
if(-not (Test-Path $log)){ 'station,expected_bytes,actual_bytes,status,seconds' | Set-Content -LiteralPath $log -Encoding UTF8 }

foreach($item in $targets){
    $dest = Join-Path $OutputDir "$($item.Station)-drvd.txt.zip"
    $url = "$base/$($item.Station)-drvd.txt.zip"
    $existing = if(Test-Path $dest){(Get-Item $dest).Length}else{0}
    if($existing -eq [int64]$item.Bytes){
        "$($item.Station),$($item.Bytes),$existing,already_complete,0" | Add-Content -LiteralPath $log -Encoding UTF8
        continue
    }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $savedErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & curl.exe --proxy $Proxy --fail --location --retry 2 --connect-timeout 20 --max-time $MaxTime --silent --show-error -C - --output $dest $url 2>&1 | Out-File -LiteralPath (Join-Path $OutputDir "$($item.Station).curl.log") -Encoding UTF8
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorAction
    $sw.Stop()
    $actual = if(Test-Path $dest){(Get-Item $dest).Length}else{0}
    $status = if($exit -eq 0 -and $actual -eq [int64]$item.Bytes){'complete'}else{"failed_$exit"}
    "$($item.Station),$($item.Bytes),$actual,$status,$([math]::Round($sw.Elapsed.TotalSeconds,1))" | Add-Content -LiteralPath $log -Encoding UTF8
    if($status -ne 'complete'){ Write-Warning "$($item.Station) $status actual=$actual expected=$($item.Bytes)" }
}
Get-Content -LiteralPath $log -Tail ($targets.Count + 1)
