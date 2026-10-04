const fs = require('fs');
const path = require('path');
const ptyPath = 'E:/vscode/Microsoft VS Code/2242ebbb54/resources/app/node_modules.asar/node-pty';
const pty = require(ptyPath);
if (process.argv.includes('--inspect')) {
  console.log('PTY module: ' + require.resolve(ptyPath));
  for (const name of ['windowsTerminal.js', 'windowsPtyAgent.js']) {
    const source = fs.readFileSync(path.join(ptyPath, 'lib', name), 'utf8');
    console.log(source.split('\n').filter(line => /conpty|useConpty|windowsPty|conptyDll|useConptyDll/i.test(line)).join('\n'));
  }
  process.exit(0);
}
const env = { ...process.env, TERM_PROGRAM: 'vscode' };
delete env.ELECTRON_RUN_AS_NODE;
const child = pty.spawn('C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe', [
  '-NoLogo', '-Command', '& "E:\\codex\\commands\\codex_au2.cmd" "Only reply OK. Do not use tools."'
], {
  name: 'xterm-256color', cols: 140, rows: 35,
  cwd: 'E:\\vscode\\cuda_proj\\SNN_proj1', env,
  useConpty: true, useConptyDll: true
});
console.log(JSON.stringify({kind: 'vscode_pty_spawn', pid: child.pid}));
const output = fs.createWriteStream(path.join(__dirname, 'vscode-pty-' + Date.now() + '.txt'), {flags: 'wx'});
child.onData(data => {
  output.write(data);
  if (data.includes('\x1b[6n')) child.write('\x1b[1;1R');
  if (data.includes('\x1b]11;?')) child.write('\x1b]11;rgb:0000/0000/0000\x1b\\');
});
child.onExit(event => { console.log(JSON.stringify({kind:'vscode_pty_exit', ...event})); output.end(); process.exit(0); });
setTimeout(() => child.write('\x03\x03'), 30000);
setTimeout(() => child.write('\x03'), 34000);
setTimeout(() => { output.end(); process.exit(0); }, 40000);
