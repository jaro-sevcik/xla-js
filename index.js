const xla = require('bindings')('xla');

const client = new xla.Client();
const b = new xla.XlaBuilder("fn")
const o = xla.constantR0f32(b, 42);
const e = b.build(o);
const l = client.compile(e, {});
const r = l.execute([], {});

const buffer = r[0][0];
const literal = buffer.toLiteralSync();

console.log("Done", literal.getFirstElementF32());

