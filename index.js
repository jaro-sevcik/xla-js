const xla = require('bindings')('xla');

const client = new xla.Client();
const b = new xla.XlaBuilder("fn")
const o = xla.constantR0f32(b, 42);

console.log("Done");

