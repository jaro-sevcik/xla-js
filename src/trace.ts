import { Shape } from "./shape";
import { Tensor } from "./tensor";

export type Mul = {
  primitive: "mul";
};

export type Add = {
  primitive: "add";
};

export type MatMul = {
  primitive: "matmul";
};

export type Transpose = {
  primitive: "transpose";
};

export type Reshape = {
  primitive: "reshape";
  shape: Shape;
};

export type Broadcast = {
  primitive: "broadcast";
  shape: Shape;
};

export type Constant = {
  primitive: "constant";
  value: Tensor;
};

export type Block = {
  primitive: "block";
};

export type Primitive =
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

export abstract class Trace<T> {
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
        value,
      } satisfies Constant,
      [],
    )[0];
  }
}

export class EvalTrace extends Trace<Tensor> {
  primitive(p: Primitive, inputs: Tensor[]): Tensor[] {
    switch (p.primitive) {
      case "constant":
        return [p.value];
      case "add":
        console.assert(inputs.length === 2);
        return [Tensor.add(inputs[0], inputs[1])];
      case "mul":
        console.assert(inputs.length === 2);
        return [Tensor.mul(inputs[0], inputs[1])];
    }

    throw new Error(`Unsupported primitive ${p}`);
  }
}
