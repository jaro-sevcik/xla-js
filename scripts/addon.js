const fs = require('fs');
const path = require('path');

{
    const name = 'xla';
    const src = path.join(__dirname, `../build/Release/${name}.node`);
    const dest = path.join(__dirname, `../xla-addon/${name}.node`);
    fs.copyFileSync(src, dest);
}
