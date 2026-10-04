@echo off
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;E:\cuda_tool_install\bin"
cd /d "E:\vscode\cuda_proj\SNN_proj1\cuda_pratice\RNSNorm"
call "E:\vs2022\VC\Auxiliary\Build\vcvars64.bat" >nul
nvcc -arch=sm_86 -O3 -o rnsnormal.exe rnsnormal.cu
echo --- 4 x 8 ---
.\rnsnormal.exe  input_4_8.txt        output_4_8.txt
.\check_rms.exe  input_4_8.txt        output_4_8.txt
echo --- 200 x 100  (N=100, padded to 104) ---
.\rnsnormal.exe  input_200_100.txt    output_200_100.txt
.\check_rms.exe  input_200_100.txt    output_200_100.txt
echo --- 64 x 4096 ---
.\rnsnormal.exe  input_64_4096.txt    output_64_4096.txt
.\check_rms.exe  input_64_4096.txt    output_64_4096.txt
echo --- 1024 x 4096 ---
.\rnsnormal.exe  input_1024_4096.txt  output_1024_4096.txt
.\check_rms.exe  input_1024_4096.txt  output_1024_4096.txt
