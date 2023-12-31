export class Client {
  constructor();
  compile(computation: XlaComputation, options: {}): PjRtLoadedExecutable;
  bufferFromHostLiteral(literal: Literal): PjRtBuffer;
}

export class XlaBuilder {
  constructor(name: string);
  build(op: XlaOp): XlaComputation;
}

export class XlaOp {}

export class XlaComputation {}

export class PjRtLoadedExecutable {
  execute(inputs: PjRtBuffer[][], options: {}): PjRtBuffer[][];
}

export class PjRtBuffer {
  toLiteralSync(): Literal;
}

export class Literal {
  getFirstElementF32(): number;
  data(ptype: PrimitiveType): number[];
  get(index: number[]): number;
  shape(): Shape;
  reshape(dimensions: number[]): Literal;
  broadcast(shape: Shape, dimensions: number[]): Literal;
  toString(): string;

  static createR0(ptype: PrimitiveType, n: number): Literal;
  static createR1(ptype: PrimitiveType, ns: number[]): Literal;
}

export enum PrimitiveType {
  F32 = "F32",
}

export class Shape {
  dimensions(): number[];
  element_type(): PrimitiveType;
  static forArray(t: PrimitiveType, dimensions: number[]): Shape;
}

export function constantR0(builder: XlaBuilder, ptype: PrimitiveType, n: number): XlaOp;
export function constantR1(builder: XlaBuilder, ptype: PrimitiveType, ns: number[]): XlaOp;
export function parameter(builder: XlaBuilder, parameter_number: number, shape: Shape, name: string): XlaOp;
export function add(lhs: XlaOp, rhs: XlaOp): XlaOp;
export function sub(lhs: XlaOp, rhs: XlaOp): XlaOp;
export function max(lhs: XlaOp, rhs: XlaOp): XlaOp;
export function mul(lhs: XlaOp, rhs: XlaOp): XlaOp;
export function div(lhs: XlaOp, rhs: XlaOp): XlaOp;
export function exp(input: XlaOp): XlaOp;
export function log(input: XlaOp): XlaOp;
export function dotGeneral(
  lhs: XlaOp,
  rhs: XlaOp,
  lhs_contracting_dimensions: number[],
  rhs_contracting_dimensions: number[],
  lhs_batch_dimensions: number[],
  rhs_batch_dimensions: number[]
): XlaOp;
export function broadcast(input: XlaOp, sizes: number[]): XlaOp;
export function broadcastInDim(input: XlaOp, sizes: number[], broadcast_dimensions: number[]): XlaOp;
export function transpose(input: XlaOp, permutation: number[]): XlaOp;
export function reshape(input: XlaOp, new_sizes: number[]): XlaOp;
export function iota(builder: XlaBuilder, shape: Shape, dimension: number): XlaOp;
export function reduce(
  builder: XlaBuilder,
  operands: XlaOp | xlaOp[],
  init_values: XlaOp | XlaOp[],
  computation: XlaComputation,
  dimensions_to_reduce: number[]
): XlaOp;

export function constantLiteral(builder: XlaBuilder, literal: Literal): XlaOp;
