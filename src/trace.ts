import { Shape } from "./shape";
import { Tensor } from "./tensor";

type Mul = {
  primitive: "mul";
};

type Add = {
  primitive: "add";
};

type MatMul = {
  primitive: "matmul";
};

type Transpose = {
  primitive: "transpose";
};

type Reshape = {
  primitive: "reshape";
  shape: Shape;
};

type Broadcast = {
  primitive: "broadcast";
  shape: Shape;
};

type Constant = {
  primitive: "constant";
  value: Tensor;
};

type Block = {
  primitive: "block";
};

type Primitive =
  | Mul
  | Add
  | MatMul
  | Transpose
  | Reshape
  | Broadcast
  | Constant
  | Block;

function assertNever(x: never): never {
  throw new Error("Unexpected value: " + x);
}

export function output_shapes(this: Primitive, input_shapes: Shape[]): Shape[] {
  switch (this.primitive) {
    case "add":
      console.assert(input_shapes.length === 2);
      console.assert(Shape.isEqual(input_shapes[0], input_shapes[1]));
      return [input_shapes[0]];
    case "mul":
      console.assert(input_shapes.length === 2);
      console.assert(Shape.isEqual(input_shapes[0], input_shapes[1]));
      return [input_shapes[0]];
    case "matmul":
      console.assert(input_shapes.length === 2);
      return [Shape.matmul(input_shapes[0], input_shapes[1])];
    case "transpose":
      console.assert(input_shapes.length === 1);
      return [Shape.transpose(input_shapes[0])];
    case "block":
      // TODO
      return [];
    case "reshape":
      console.assert(input_shapes.length === 1);
      console.assert(input_shapes[0].total_size() === this.shape.total_size());
      return [this.shape];
    case "broadcast":
      console.assert(input_shapes.length === 1);
      console.assert(input_shapes[0].total_size() === this.shape.total_size());
      return [this.shape];
    case "constant":
      return [this.value.shape()];
  }
  assertNever(this);
}

abstract class Trace<T> {
  abstract primitive(p: Primitive, inputs: T[]): T[];

  mul(lhs: T, rhs: T): T {
    return this.primitive(
      {
        primitive: "mul",
      } satisfies Mul,
      [lhs, rhs],
    )[0];
  }

  add(lhs: T, rhs: T): T {
    return this.primitive(
      {
        primitive: "add",
      } satisfies Add,
      [lhs, rhs],
    )[0];
  }

  constant(value: Tensor) {
    return this.primitive(
      {
        primitive: "constant",
        value
      } satisfies Constant,
      [],
    )[0];
  }
}

function myfn<T>(trace: Trace<T>, x: T) {
  const exprX2 = trace.mul(x, x);
  const expr4X = trace.mul(trace.constant(Tensor.constantR0(4)), x);
  return trace.add(exprX2, trace.add(expr4X, trace.constant(Tensor.constantR0(6))));
}
