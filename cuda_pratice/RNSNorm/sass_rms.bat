@echo off
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;E:\cuda_tool_install\bin;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.1"
cd /d "E:\vscode\cuda_proj\SNN_proj1\cuda_pratice\RNSNorm"
call "E:\vs2022\VC\Auxiliary\Build\vcvars64.bat" >nul
nvcc -arch=sm_86 -O3 -Xptxas -v -cubin -o rnsnormal.cubin rnsnormal.cu 2>&1 | findstr /C:"Used"
cuobjdump -sass rnsnormal.cubin > rnsnormal.sass
ncu --clock-control base --metrics sm__cycles_elapsed.avg,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed --csv .\rnsnormal.exe input_1024_4096.txt output_tmp.txt > rms_prof.csv 2>&1
