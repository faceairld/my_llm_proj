import ctypes
import importlib.util
import json
import os
import pathlib
import sqlite3
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent

if len(sys.argv) < 2 or sys.argv[1] == 'inspect':
    print({m: bool(importlib.util.find_spec(m)) for m in ['psutil', 'win32com', 'win32process', 'win32gui', 'winpty']})
    c = sqlite3.connect('file:E:/codex/home-au2/logs_2.sqlite?mode=ro', uri=True)
    print(c.execute("SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall())
    import inspect
    from winpty import PtyProcess
    print(inspect.signature(PtyProcess.spawn))
    print(c.execute("SELECT datetime(ts,'unixepoch'), level, target, substr(feedback_log_body,1,900) FROM logs WHERE level IN ('ERROR','WARN') ORDER BY id DESC LIMIT 12").fetchall())
    sys.exit(0)

import psutil
import threading
from ctypes import wintypes

u = ctypes.windll.user32
u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
u.IsWindowVisible.argtypes = [wintypes.HWND]
CALLBACK = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
u.EnumWindows.argtypes = [CALLBACK, wintypes.LPARAM]

targets = {'codex.exe', 'conhost.exe', 'openconsole.exe', 'cmd.exe', 'powershell.exe', 'pwsh.exe', 'node_repl.exe', 'codex-code-mode-host.exe', 'codex-computer-use.exe', 'git.exe', 'rg.exe', 'where.exe', 'whoami.exe', 'codex-command-runner.exe', 'codex-windows-sandbox-setup.exe'}
known = {}
last_windows = {}
start = time.monotonic()
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 45
out = BASE / ('trace-' + time.strftime('%H%M%S') + '.jsonl')
with out.open('x', encoding='utf-8') as f:
    write_lock = threading.Lock()
    def emit(event):
        event['elapsed'] = round(time.monotonic() - start, 4)
        with write_lock:
            if f.closed:
                return
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
            f.flush()
        if event['kind'] != 'initial_process':
            print(json.dumps(event, ensure_ascii=False), flush=True)

    def window_events():
        PROC = ctypes.WINFUNCTYPE(None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND, ctypes.c_long, ctypes.c_long, wintypes.DWORD, wintypes.DWORD)
        u.SetWinEventHook.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.HMODULE, PROC, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
        u.SetWinEventHook.restype = wintypes.HANDLE
        @PROC
        def on_event(hook, event, hwnd, obj, child_id, thread_id, timestamp):
            if not hwnd or obj != 0 or child_id != 0:
                return
            cls = ctypes.create_unicode_buffer(256)
            u.GetClassNameW(hwnd, cls, 256)
            if cls.value not in ('ConsoleWindowClass', 'CASCADIA_HOSTING_WINDOW_CLASS'):
                return
            pid = wintypes.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            title = ctypes.create_unicode_buffer(1024)
            u.GetWindowTextW(hwnd, title, 1024)
            emit(dict(kind='window_event', event=event, hwnd=hwnd, pid=pid.value, title=title.value, window_class=cls.value, visible=bool(u.IsWindowVisible(hwnd))))
        hook = u.SetWinEventHook(0x8000, 0x8003, None, on_event, 0, 0, 0)
        emit(dict(kind='window_hook', active=bool(hook)))
        msg = wintypes.MSG()
        while u.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            u.TranslateMessage(ctypes.byref(msg))
            u.DispatchMessageW(ctypes.byref(msg))
    threading.Thread(target=window_events, daemon=True).start()

    def scan_processes(initial=False):
        for p in psutil.process_iter(['pid', 'ppid', 'name', 'create_time']):
            info = p.info
            if (info['name'] or '').lower() not in targets:
                continue
            key = (info['pid'], info['create_time'])
            if key in known:
                continue
            try:
                info['exe'] = p.exe()
                info['cmdline'] = p.cmdline()
            except (psutil.Error, OSError):
                pass
            known[key] = info
            emit(dict(kind='initial_process' if initial else 'process_start', **info))

    def scan_windows(initial=False):
        windows = {}
        @CALLBACK
        def enum(hwnd, _):
            if not u.IsWindowVisible(hwnd):
                return True
            cls = ctypes.create_unicode_buffer(256)
            u.GetClassNameW(hwnd, cls, 256)
            if cls.value not in ('ConsoleWindowClass', 'CASCADIA_HOSTING_WINDOW_CLASS'):
                return True
            pid = wintypes.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            title = ctypes.create_unicode_buffer(1024)
            u.GetWindowTextW(hwnd, title, 1024)
            info = dict(hwnd=hwnd, pid=pid.value, window_class=cls.value, title=title.value)
            windows[hwnd] = info
            if hwnd not in last_windows:
                try:
                    p = psutil.Process(pid.value)
                    info.update(exe=p.exe(), cmdline=p.cmdline(), ppid=p.ppid())
                except (psutil.Error, OSError):
                    pass
                emit(dict(kind='initial_window' if initial else 'window_visible', **info))
            return True
        u.EnumWindows(enum, 0)
        for hwnd, info in last_windows.items():
            if hwnd not in windows:
                emit(dict(kind='window_gone', **info))
        last_windows.clear()
        last_windows.update(windows)

    scan_processes(True)
    scan_windows(True)
    print('READY ' + str(out), flush=True)
    child = None
    node_child = None
    if sys.argv[1] == 'reproduce-vscode':
        import subprocess
        node_env = dict(os.environ, ELECTRON_RUN_AS_NODE='1')
        node_log = (BASE / ('vscode-host-' + time.strftime('%H%M%S') + '.txt')).open('x', encoding='utf-8')
        node_child = subprocess.Popen([
            r'E:\vscode\Microsoft VS Code\Code.exe', str(BASE / 'vscode-pty.cjs')
        ], env=node_env, stdout=node_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        emit(dict(kind='vscode_host_spawn', pid=node_child.pid))
    if sys.argv[1] == 'reproduce':
        from winpty import PtyProcess
        command = 'cmd.exe /d /c E:\\codex\\commands\\codex_au2.cmd "Only reply OK. Do not use tools."'
        child = PtyProcess.spawn(command, dimensions=(35, 140), cwd=r'E:\vscode\cuda_proj\SNN_proj1', backend=0)
        emit(dict(kind='probe_spawn', pid=child.pid, command=command))
        def read_child():
            with (BASE / ('pty-' + time.strftime('%H%M%S') + '.txt')).open('x', encoding='utf-8') as transcript:
                try:
                    while child.isalive():
                        data = child.read(8192)
                        transcript.write(data)
                        transcript.flush()
                        if '\x1b[6n' in data:
                            child.write('\x1b[1;1R')
                        if '\x1b]11;?' in data:
                            child.write('\x1b]11;rgb:0000/0000/0000\x1b\\')
                except (EOFError, OSError):
                    pass
        threading.Thread(target=read_child, daemon=True).start()
    sent_exit = False
    while time.monotonic() - start < duration:
        scan_windows()
        scan_processes()
        if child is not None and not sent_exit and time.monotonic() - start > duration - 8:
            child.write('\x03\x03')
            sent_exit = True
            emit(dict(kind='probe_ctrl_c', pid=child.pid))
        time.sleep(0.01)
    emit(dict(kind='done', output=str(out)))
