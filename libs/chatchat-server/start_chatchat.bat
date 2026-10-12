@echo off
chcp 65001 >nul
cd /d "%~dp0"
set CHATCHAT_ROOT=C:\Users\Lenovo\chatchat
rem 中文路径下必须开启 Python UTF-8 模式，否则 .pth 文件按 GBK 解码会报错
set PYTHONUTF8=1
.venv\Scripts\python.exe chatchat\cli.py start -a
pause