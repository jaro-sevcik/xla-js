export class Client {
  constructor();
  compile(computation: XlaComputation, options: {}): PjRtLoadedExecutable;
  bufferFromHostLiteral(literal: Literal): PjRtBuffer;
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
  execute(inputs: PjRtBuffer[][], options: {}): PjRtBuffer[][];
}

export class PjRtBuffer {
  toLiteralSync(): Literal;
}

export class Literal {
  getFirstElementF32(): number;

  static createR0(ptype: PrimitiveType, n: number): Literal;
}

enum PrimitiveType {
  F32 = "F32",
}

export class Shape {
  dimensions(): number[];
  element_type(): PrimitiveType;
  static forArray(t: PrimitiveType, dimensions: number[]): Shape;
}

export function constantR0f32(builder: XlaBuilder, n: number): XlaOp;
export function parameter(builder: XlaBuilder, parameter_number: number, shape: Shape, name: string): XlaOp;
export function add(lhs: XlaOp, rhs: XlaOp): XlaOp;