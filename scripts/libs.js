const paths = [];

paths.push(`-Wl,-rpath,'$$ORIGIN/../../xla_extension/lib'`);
paths.push(`-Wl,-rpath,'$$ORIGIN/../xla_extension/lib'`);
paths.push(`-L./../xla_extension/lib`)
paths.push(`-lxla_extension`);

console.log(paths.join(' '));