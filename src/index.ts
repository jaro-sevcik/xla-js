import * as xla from '../xla-addon';

const client = new xla.Client();

// Build the computation.
const b = new xla.XlaBuilder("fn");
const shape = xla.Shape.forArray(xla.PrimitiveType.F32, []);
const o = xla.constantR0f32(b, 42);
const p = xla.parameter(b, 0, shape, "x");
const e = b.build(xla.add(o, p));

// Compile the computation.
const l = client.compile(e, {});

// Execute the computation.
const i = xla.Literal.createR0(xla.PrimitiveType.F32, 52);
const r = l.execute([[client.bufferFromHostLiteral(i)]], {});

const buffer = r[0][0];
const literal = buffer.toLiteralSync();

console.log("Done", literal.getFirstElementF32());

const l2 = xla.Literal.createR0(xla.PrimitiveType.F32, 52);
console.log("New lit", l2.getFirstElementF32());
