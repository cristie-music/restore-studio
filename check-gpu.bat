@echo off
rem Shows which device Restore Studio will use (GPU or CPU) and why.
setlocal
chcp 65001 >nul
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
"%~dp0python\python.exe" -s "%~dp0app\studio\device.py"
echo.
nvidia-smi 2>nul || echo nvidia-smi не найден: драйвер NVIDIA не установлен (или видеокарта не NVIDIA) - будет CPU.
echo.
echo Нужен драйвер NVIDIA 560+ для CUDA 12.6. Без видеокарты всё работает на процессоре (медленнее).
pause
