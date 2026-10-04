@echo off
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;E:\cuda_tool_install\bin"
cd /d "E:\vscode\cuda_proj\SNN_proj1\cuda_pratice\Transposition"
for %%s in (4_6 32_32 200_100 100_200 1_97 97_1 1000_33 513_1027 1027_4099) do (
  echo ===== %%s =====
  .\trans.exe input_%%s.txt output_%%s.txt
  .\check_trans.exe input_%%s.txt output_%%s.txt
)
