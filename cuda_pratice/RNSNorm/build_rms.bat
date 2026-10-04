@echo off
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;E:\cuda_tool_install\bin"
cd /d "E:\vscode\cuda_proj\SNN_proj1\cuda_pratice\RNSNorm"
call "E:\vs2022\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc gen_rms.cpp /Fe:gen_rms.exe
cl /nologo /O2 /EHsc check_rms.cpp /Fe:check_rms.exe
.\gen_rms.exe    4  8  1 input_4_8.txt
.\gen_rms.exe  200 100 2 input_200_100.txt
.\gen_rms.exe   64 4096 3 input_64_4096.txt
.\gen_rms.exe 1024 4096 4 input_1024_4096.txt
