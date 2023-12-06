
const libPaths = [
    '/home/jarin/projects/xla-js/xla_extension/lib',
];
const xlaLibList = [
    "xla_extension",
].map(x => `-l${x}`);

const paths = [];
libPaths.forEach(libPath => {
    paths.push(`-L${libPath}`);
    paths.push(`-Wl,-rpath,'$$ORIGIN/../../xla_extension/lib'`);
});

paths.push(...xlaLibList);

console.log(paths.join(' '));