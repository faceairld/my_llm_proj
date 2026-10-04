const fs = require('fs');
const roots = ['E:/vscode/Microsoft VS Code/2242ebbb54/resources/app/out/vs/workbench/workbench.desktop.main.js'];
for(const file of roots){
 if(!fs.existsSync(file)){console.log('missing '+file);continue;}
 const src=fs.readFileSync(file,'utf8');
 for(const term of ['enableImages', 'allowTransparency', 'raw.options.allowTransparency']){
  let pos=0,count=0;
  while((pos=src.indexOf(term,pos))>=0&&count++<12){console.log(term+': '+src.slice(Math.max(0,pos-130),pos+480));pos+=term.length;}
 }
}
