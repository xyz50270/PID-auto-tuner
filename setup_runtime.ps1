# setup_runtime.ps1
$PythonVersion = "3.10.11"
$PythonZip = "python-$PythonVersion-embed-amd64.zip"
$DownloadUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonZip"
$RuntimeDir = Join-Path $PSScriptRoot "runtime"
$PipUrl = "https://bootstrap.pypa.io/get-pip.py"

if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}

Write-Host "Downloading Python $PythonVersion embeddable package..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $DownloadUrl -OutFile "$RuntimeDir\$PythonZip"

Write-Host "Extracting Python..." -ForegroundColor Cyan
Expand-Archive -Path "$RuntimeDir\$PythonZip" -DestinationPath $RuntimeDir -Force
Remove-Item "$RuntimeDir\$PythonZip"

# Configure .pth file to enable site-packages
$PthFile = Get-ChildItem -Path $RuntimeDir -Filter "python*._pth" | Select-Object -First 1
if ($PthFile) {
    Write-Host "Configuring $($PthFile.Name) to enable site-packages..." -ForegroundColor Cyan
    $Content = Get-Content $PthFile.FullName
    $Content = $Content -replace "#import site", "import site"
    $Content | Out-File -FilePath $PthFile.FullName -Encoding ascii
}

# Install pip
Write-Host "Downloading get-pip.py..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $PipUrl -OutFile "$RuntimeDir\get-pip.py"

Write-Host "Installing pip..." -ForegroundColor Cyan
Start-Process -FilePath "$RuntimeDir\python.exe" -ArgumentList "$RuntimeDir\get-pip.py" -Wait -NoNewWindow

# Install dependencies
if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    Start-Process -FilePath "$RuntimeDir\python.exe" -ArgumentList "-m pip install -r requirements.txt" -Wait -NoNewWindow
}

Write-Host "Setup complete! Use run.bat to start the application." -ForegroundColor Green
