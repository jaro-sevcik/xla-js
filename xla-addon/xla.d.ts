export class Client {
  constructor();
  compile(computation: XlaComputation, options: {}): PjRtLoadedExecutable;
}

export class XlaBuilder {
  constructor(name: string);
  build(op: XlaOp): XlaComputation;
}

export class XlaOp {
}

export class XlaComputation {
}

export class PjRtLoadedExecutable {
  execute(inputs: Literal[], options: {}): PjRtBuffer[][];
}

export class PjRtBuffer {
  toLiteralSync(): Literal;
}

export class Literal {
  getFirstElementF32(): number;
}

export function constantR0f32(builder: XlaBuilder, n: number): XlaOp;