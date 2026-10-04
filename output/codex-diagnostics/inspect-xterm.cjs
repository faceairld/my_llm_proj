const fs = require('fs');
const base = 'E:/vscode/Microsoft VS Code/2242ebbb54/resources/app/node_modules.asar/';
for (const rel of ['@xterm/addon-webgl/lib/addon-webgl.js','@xterm/xterm/lib/xterm.js']) {
  const full = base + rel;
  if (!fs.existsSync(full)) { console.log('Missing ' + full); continue; }
  const src = fs.readFileSync(full, 'utf8');
  console.log('\nFILE ' + rel);
  for (const term of ['_resolveForegroundRgba(', '_getForegroundColor(', '_getMinimumContrastColor(']) {
    let start = 0, count = 0;
    while ((start = src.indexOf(term, start)) >= 0 && count++ < 3) {
      console.log(term + ': ' + src.slice(Math.max(0,start-150),start+1300));
      start += term.length;
    }
  }
}
