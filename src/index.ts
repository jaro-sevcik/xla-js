import * as xla from '../xla-addon';

const client = new xla.Client();

{
    // Build the computation.
    const b = new xla.XlaBuilder("fn");
    const shape = xla.Shape.forArray(xla.PrimitiveType.F32, []);
    const o = xla.constantR0(b, xla.PrimitiveType.F32, 42);
    const p = xla.parameter(b, 0, shape, "x");
    const e = b.build(xla.add(o, p));

    // Compile the computation.
    const l = client.compile(e, {});

    // Execute the computation.
    const i = xla.Literal.createR0(xla.PrimitiveType.F32, 52);
    const r = l.execute([[client.bufferFromHostLiteral(i)]], {});

    const buffer = r[0][0];
    const literal = buffer.toLiteralSync();
    console.log("Scalar: ", literal.getFirstElementF32());
}

{
    // Build the computation.
    const b = new xla.XlaBuilder("fn");
    const shape = xla.Shape.forArray(xla.PrimitiveType.F32, [2]);
    const o = xla.constantR1(b, xla.PrimitiveType.F32, [41, 42]);
    const p = xla.parameter(b, 0, shape, "x");
    const e = b.build(xla.add(o, p));

    // Compile the computation.
    const l = client.compile(e, {});

    // Execute the computation.
    const i = xla.Literal.createR1(xla.PrimitiveType.F32, [1, 2]);
    const r = l.execute([[client.bufferFromHostLiteral(i)]], {});

    const buffer = r[0][0];
    const literal = buffer.toLiteralSync();
    console.log("Scalar: ", literal.toString());
    console.log("Data: ", literal.data(xla.PrimitiveType.F32));
    console.log("Shape: ", literal.shape().dimensions());
}
