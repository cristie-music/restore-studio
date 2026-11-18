@echo off
rem Restore Studio without the browser:  restore.bat "D:\music\song.flac" [--bpm 120] [--model sw6|rofo4] [--quality high|normal|fast] [--device auto|cuda|cpu]
rem Drag an audio file onto this .bat to process it. Results: projects\<date>_<name>\
setlocal
chcp 65001 >nul
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
if "%~1"=="" (
  set /p "IN=Перетащите сюда аудиофайл и нажмите Enter: "
) else (
  set "IN=%~1"
)
set "IN=%IN:"=%"
"%~dp0python\python.exe" -s "%~dp0app\restore.py" "%IN%" %2 %3 %4 %5 %6 %7 %8 %9
pause
