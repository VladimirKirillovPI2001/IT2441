# Сборка exe для релиза. Запуск: .\build_exe.ps1
# Требуется: установленный venv с зависимостями + pyinstaller

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "venv\Scripts\activate.ps1")) {
    Write-Host "Создайте venv и установите зависимости: python -m venv venv; .\venv\Scripts\activate; pip install -r requirements.txt"
    exit 1
}

& .\venv\Scripts\pip.exe install pyinstaller --quiet
& .\venv\Scripts\pyinstaller.exe build_exe.spec --noconfirm --clean

if (Test-Path "dist\PlantClassification\PlantClassification.exe") {
    Write-Host ""
    Write-Host "Готово. Exe и файлы: dist\PlantClassification\"
    Write-Host "Для релиза: заархивируйте папку dist\PlantClassification и загрузите zip в Assets релиза на GitHub."
}
