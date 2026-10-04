@echo off
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem"
cd /d "E:\vscode\cuda_proj\SNN_proj1\cuda_pratice\RNSNorm"
.\check_rms.exe input_1024_4096.txt output_1024_4096_2.txt
