@echo off
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;E:\cuda_tool_install\bin"
cd /d "E:\vscode\cuda_proj\SNN_proj1\cuda_pratice\Transposition"
call "E:\vs2022\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc gen_trans.cpp /Fe:gen_trans.exe
cl /nologo /O2 /EHsc check_trans.cpp /Fe:check_trans.exe
.\gen_trans.exe    4    6 1 input_4_6.txt
.\gen_trans.exe   32   32 2 input_32_32.txt
.\gen_trans.exe  200  100 3 input_200_100.txt
.\gen_trans.exe  100  200 4 input_100_200.txt
.\gen_trans.exe    1   97 5 input_1_97.txt
.\gen_trans.exe   97    1 6 input_97_1.txt
.\gen_trans.exe 1000   33 7 input_1000_33.txt
.\gen_trans.exe  513 1027 8 input_513_1027.txt
.\gen_trans.exe 1027 4099 9 input_1027_4099.txt
