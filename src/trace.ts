import * as Permutation from "./permutation";
import { DotGeneralDimensions, Shape } from "./shape";
import { Tensor, TensorLiteral } from "./tensor";
import { strict as assert } from "assert";

export interface Shaped {
  shape(): Shape;
}

export type Mul = {
  primitive: "mul";
};

export type Add = {
  primitive: "add";
};

export type DotGeneral = {
  primitive: "dotGeneral";
  dimensions: DotGeneralDimensions;
};

export type Transpose = {
  primitive: "transpose";
  permutation: number[];
};

export type Reshape = {
  primitive: "reshape";
  new_sizes: number[];
};

export type Broadcast = {
  primitive: "broadcast";
  new_sizes: number[];
  broadcast_dimensions: number[];
};

export type ReduceSum = {
  primitive: "reduceSum";
  axes: number[];
};

export type Constant = {
  primitive: "constant";
  value: Tensor;
};

export type Block = {
  primitive: "block";
};

export type Primitive = Mul | Add | DotGeneral | Transpose | Reshape | Broadcast | ReduceSum | Constant | Block;

function assertNever(x: never): never {
  throw new Error("Unexpected value: " + x);
}

export function output_shapes(this: Primitive, input_shapes: Shape[]): Shape[] {
  switch (this.primitive) {
    case "add":
      assert.strictEqual(input_shapes.length, 2);
      assert.ok(Shape.isEqual(input_shapes[0], input_shapes[1]));
      return [input_shapes[0]];
    case "mul":
      assert.strictEqual(input_shapes.length, 2);
      assert.ok(Shape.isEqual(input_shapes[0], input_shapes[1]));
      return [input_shapes[0]];
    case "dotGeneral":
      assert.strictEqual(input_shapes.length, 2);
      return [Shape.dotGeneral(input_shapes[0], input_shapes[1], this.dimensions)];
    case "transpose":
      assert.strictEqual(input_shapes.length, 1);
      return [input_shapes[0].transpose(this.permutation)];
    case "block":
      // TODO
      return [];
    case "reshape":
      assert.strictEqual(input_shapes.length, 1);
      assert.strictEqual(
        input_shapes[0].total_size(),
        this.new_sizes.reduce((s, v) => s * v, 1),
      );
      return [new Shape(input_shapes[0].element_type(), this.new_sizes)];
    case "broadcast":
      assert.strictEqual(input_shapes.length, 1);
      assert.strictEqual(input_shapes[0].rank(), this.new_sizes);
      for (let i = 0; i < this.new_sizes.length; i++) {
        assert.ok(input_shapes[0].dimensions()[i] === 1 || input_shapes[0].dimensions()[i] === this.new_sizes[i]);
      }
      return [new Shape(input_shapes[0].element_type(), this.new_sizes)];
    case "constant":
      assert.strictEqual(input_shapes.length, 0);
      return [this.value.shape()];
    case "reduceSum": {
      assert.strictEqual(input_shapes.length, 1);
      return [input_shapes[0].removeAxes(this.axes)];
    }
  }
  assertNever(this);
}

export abstract class Trace<T extends Shaped> {
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

  constant(value: Tensor): T {
    return this.primitive(
      {
        primitive: "constant",
        value,
      } satisfies Constant,
      [],
    )[0];
  }

  literal(value: TensorLiteral): T {
    return this.primitive(
      {
        primitive: "constant",
        value: Tensor.literal(value),
      } satisfies Constant,
      [],
    )[0];
  }

  dotGeneral(lhs: T, rhs: T, dimensions: DotGeneralDimensions): T {
    return this.primitive(
      {
        primitive: "dotGeneral",
        dimensions,
      } satisfies DotGeneral,
      [lhs, rhs],
    )[0];
  }

  reduceSum(input: T, axes: number[]): T {
    return this.primitive(
      {
        primitive: "reduceSum",
        axes,
      } satisfies ReduceSum,
      [input],
    )[0];
  }

  matmul(lhs: T, rhs: T): T {
    const rank = lhs.shape().rank();
    assert.ok(rank >= 2);
    assert.strictEqual(rank, rhs.shape().rank());
    const batch = new Array(rank - 2).fill(0).map((e, i) => i);
    return this.dotGeneral(lhs, rhs, {
      contracting_lhs: [rank - 1],
      contracting_rhs: [rank - 2],
      batch_lhs: batch,
      batch_rhs: batch,
    });
  }

  transpose(input: T, permutation: number[]): T {
    return this.primitive(
      {
        primitive: "transpose",
        permutation,
      } satisfies Transpose,
      [input],
    )[0];
  }

  reshape(input: T, new_sizes: number[]): T {
    return this.primitive(
      {
        primitive: "reshape",
        new_sizes,
      } satisfies Reshape,
      [input],
    )[0];
  }

  broadcast(input: T, new_sizes: number[], broadcast_dimensions?: number[]): T {
    if (!broadcast_dimensions) {
      broadcast_dimensions = range(0, new_sizes.length);
    }
    return this.primitive(
      {
        primitive: "broadcast",
        new_sizes,
        broadcast_dimensions,
      } satisfies Broadcast,
      [input],
    )[0];
  }
}

export class EvalTrace extends Trace<Tensor> {
  primitive(p: Primitive, inputs: Tensor[]): Tensor[] {
    switch (p.primitive) {
      case "constant":
        return [p.value];
      case "add":
        assert.strictEqual(inputs.length, 2);
        return [Tensor.add(inputs[0], inputs[1])];
      case "mul":
        assert.strictEqual(inputs.length, 2);
        return [Tensor.mul(inputs[0], inputs[1])];
      case "dotGeneral":
        assert.strictEqual(inputs.length, 2);
        return [Tensor.dotGeneral(inputs[0], inputs[1], p.dimensions)];
      case "transpose":
        assert.strictEqual(inputs.length, 1);
        return [inputs[0].transpose(p.permutation)];
      case "reshape":
        assert.strictEqual(inputs.length, 1);
        return [inputs[0].reshape(p.new_sizes)];
      case "broadcast":
        assert.strictEqual(inputs.length, 1);
        return [inputs[0].broadcastInDim(p.new_sizes, p.broadcast_dimensions)];
      case "reduceSum":
        assert.strictEqual(inputs.length, 1);
        return [inputs[0].reduceSum(p.axes)];
      case "block":
        throw new Error(`Block not implemented`);
    }
    assertNever(p);
  }
}

type LinearExpressionInput = { kind: "input"; index: number };
type LinearExpressionIndex = { kind: "expr"; index: number };
type LinearExpressionZero = { kind: "zero"; shape: Shape };
type LinearExpressionValue = LinearExpressionInput | LinearExpressionIndex | LinearExpressionZero;

type LinearExpressionMul<T> = {
  kind: "mul";
  lhs: LinearExpressionValue;
  rhs: T;
};
type LinearExpressionAdd = {
  kind: "add";
  lhs: LinearExpressionValue;
  rhs: LinearExpressionValue;
};
type LinearExpressionDotLeft<T> = {
  kind: "dotGeneralLeft";
  lhs: LinearExpressionValue;
  rhs: T;
  dimensions: DotGeneralDimensions;
};
type LinearExpressionDotRight<T> = {
  kind: "dotGeneralRight";
  lhs: T;
  rhs: LinearExpressionValue;
  dimensions: DotGeneralDimensions;
};
type LinearExpressionTranspose = {
  kind: "transpose";
  input: LinearExpressionValue;
  permutation: number[];
};
type LinearExpressionBroadcast = {
  kind: "broadcast";
  input: LinearExpressionValue;
  sizes: number[];
  broadcastDimensions: number[];
};
type LinearExpressionReduceSum = {
  kind: "reduceSum";
  input: LinearExpressionValue;
  axes: number[];
};

type LinearExpression<T> =
  | LinearExpressionMul<T>
  | LinearExpressionAdd
  | LinearExpressionDotLeft<T>
  | LinearExpressionDotRight<T>
  | LinearExpressionTranspose
  | LinearExpressionBroadcast
  | LinearExpressionReduceSum;

class LinearGraph<T> {
  expressions: { shape: Shape; expression: LinearExpression<T> }[] = [];

  constructor(readonly input_shapes: Shape[]) {}
}

class LinearExpressionEvaluationContext<T extends Shaped> {
  values: T[];
  inputs: T[];
  #trace: Trace<T>;

  constructor(input_shapes: Shape[], expression_shapes: Shape[], trace: Trace<T>) {
    this.values = expression_shapes.map((s) => trace.constant(Tensor.zeros(s)));
    this.inputs = input_shapes.map((s) => trace.constant(Tensor.zeros(s)));
    this.#trace = trace;
  }

  addToValue(index: LinearExpressionValue, value: T) {
    switch (index.kind) {
      case "expr":
        this.values[index.index] = this.#trace.add(this.values[index.index], value);
        break;
      case "input":
        this.inputs[index.index] = this.#trace.add(this.inputs[index.index], value);
        break;
      case "zero":
        break;
    }
  }

  shape(index: LinearExpressionValue): Shape {
    switch (index.kind) {
      case "expr":
        return this.values[index.index].shape();
      case "input":
        return this.inputs[index.index].shape();
      case "zero":
        return index.shape;
    }
  }
}

class GradTracer<T extends Shaped> implements Shaped {
  constructor(
    readonly value: T,
    readonly grad: LinearExpressionValue,
  ) {}

  shape(): Shape {
    return this.value.shape();
  }
}

function range(from: number, size: number) {
  const result = new Array(size);
  for (let i = 0; i < size; i++) {
    result[i] = i + from;
  }
  return result;
}

type DotGeneralFullDimensions = {
  contracting_lhs: number[];
  contracting_rhs: number[];
  batch_lhs: number[];
  batch_rhs: number[];
  other_lhs: number[];
  other_rhs: number[];
};

function dotGeneralFullDims(
  { contracting_lhs, contracting_rhs, batch_lhs, batch_rhs }: DotGeneralDimensions,
  lhs_rank: number,
  rhs_rank: number,
): DotGeneralFullDimensions {
  // Compute the remaining dims (after removing batch and contracting dims) for lhs and rhs.
  let other_rhs = range(0, rhs_rank);
  for (const i of contracting_rhs) other_rhs[i] = -1;
  for (const i of batch_rhs) other_rhs[i] = -1;
  other_rhs = other_rhs.filter((i) => i >= 0);

  let other_lhs = range(0, lhs_rank);
  for (const i of contracting_lhs) other_lhs[i] = -1;
  for (const i of batch_lhs) other_lhs[i] = -1;
  other_lhs = other_lhs.filter((i) => i >= 0);

  return { contracting_lhs, contracting_rhs, batch_lhs, batch_rhs, other_lhs, other_rhs };
}

class GradTrace<T extends Shaped> extends Trace<GradTracer<T>> {
  #inner: Trace<T>;
  #linearGraph: LinearGraph<T>;

  constructor(inner: Trace<T>, input_shapes: Shape[]) {
    super();
    this.#inner = inner;
    this.#linearGraph = new LinearGraph<T>(input_shapes);
  }

  addExpression(shape: Shape, expression: LinearExpression<T>): number {
    this.#linearGraph.expressions.push({ shape, expression });
    return this.#linearGraph.expressions.length - 1;
  }

  linearAdd(lhs: LinearExpressionValue, rhs: LinearExpressionValue, shape: Shape): LinearExpressionValue {
    if (lhs.kind === "zero") return rhs;
    if (rhs.kind === "zero") return lhs;
    const index = this.addExpression(shape, { kind: "add", lhs, rhs });
    return { kind: "expr", index };
  }

  linearMul(lhs: LinearExpressionValue, rhs: T, shape: Shape): LinearExpressionValue {
    if (lhs.kind === "zero") return lhs;
    const index = this.addExpression(shape, { kind: "mul", lhs, rhs });
    return { kind: "expr", index };
  }

  linearTranspose(input: LinearExpressionValue, permutation: number[], shape: Shape): LinearExpressionValue {
    if (input.kind === "zero") return { kind: "zero", shape };
    const index = this.addExpression(shape, { kind: "transpose", input, permutation });
    return { kind: "expr", index };
  }

  linearBroadcast(
    input: LinearExpressionValue,
    new_sizes: number[],
    broadcast_dimensions: number[],
    shape: Shape,
  ): LinearExpressionValue {
    if (input.kind === "zero") return { kind: "zero", shape };
    const index = this.addExpression(shape, {
      kind: "broadcast",
      input,
      sizes: new_sizes,
      broadcastDimensions: broadcast_dimensions,
    });
    return { kind: "expr", index };
  }

  linearReduceSum(input: LinearExpressionValue, axes: number[], shape: Shape): LinearExpressionValue {
    if (input.kind === "zero") return { kind: "zero", shape };
    const index = this.addExpression(shape, { kind: "reduceSum", input, axes });
    return { kind: "expr", index };
  }

  linearDotGeneralLeft(
    lhs: LinearExpressionValue,
    rhs: T,
    dimensions: DotGeneralDimensions,
    shape: Shape,
  ): LinearExpressionValue {
    if (lhs.kind === "zero") return lhs;
    const index = this.addExpression(shape, {
      kind: "dotGeneralLeft",
      lhs,
      rhs,
      dimensions,
    });
    return { kind: "expr", index };
  }

  linearDotGeneralRight(
    lhs: T,
    rhs: LinearExpressionValue,
    dimensions: DotGeneralDimensions,
    shape: Shape,
  ): LinearExpressionValue {
    if (rhs.kind === "zero") return rhs;
    const index = this.addExpression(shape, {
      kind: "dotGeneralRight",
      lhs,
      rhs,
      dimensions,
    });
    return { kind: "expr", index };
  }

  linearZero(shape: Shape): LinearExpressionValue {
    return { kind: "zero", shape };
  }

  evaluateGraphBackwards(from: LinearExpressionValue): T[] {
    const expressions = this.#linearGraph.expressions;
    const expression_shapes = expressions.map(({ shape }) => shape);
    const context = new LinearExpressionEvaluationContext(
      this.#linearGraph.input_shapes,
      expression_shapes,
      this.#inner,
    );

    context.addToValue(from, this.#inner.constant(Tensor.constantR0(1.0)));

    for (let position = expressions.length - 1; position >= 0; position--) {
      const value = context.values[position];
      const e = expressions[position].expression;
      switch (e.kind) {
        case "add":
          context.addToValue(e.lhs, value);
          context.addToValue(e.rhs, value);
          break;
        case "mul":
          context.addToValue(e.lhs, this.#inner.mul(value, e.rhs));
          break;
        case "transpose": {
          context.addToValue(e.input, this.#inner.transpose(value, Permutation.invert(e.permutation)));
          break;
        }
        case "broadcast": {
          throw new Error("Not implemented");
        }
        case "reduceSum": {
          const sizes = context.shape(e.input).dimensions();
          const dims: number[] = [];
          let finger = 0;
          for (let i = 0; i < sizes.length; i++) {
            if (finger < e.axes.length && e.axes[finger] === i) {
              finger++; // This is a summed dimension, let us skip it in broadcasted dimensions.
            } else {
              dims.push(i);
            }
          }
          context.addToValue(e.input, this.#inner.broadcast(value, sizes, dims));
          break;
        }
        case "dotGeneralLeft": {
          const { contracting_lhs, contracting_rhs, batch_lhs, batch_rhs, other_lhs, other_rhs } = dotGeneralFullDims(
            e.dimensions,
            value.shape().rank(),
            e.rhs.shape().rank(),
          );

          // Compute the dimensions for the trannsposed product.
          const transposed_dims = {
            contracting_lhs: range(batch_lhs.length + other_lhs.length, other_rhs.length),
            contracting_rhs: other_rhs,
            batch_lhs: range(0, batch_lhs.length),
            batch_rhs,
          };
          const transposed_product = this.#inner.dotGeneral(value, e.rhs, transposed_dims);

          // Transposed product starts with batch dims, then lhs other dims and finally the contracting dims.
          // Let us shuffle them around. Note that the contracting dims are in the rhs order, so we first
          // need to permute the contracting dims to the index order and to the lhs order.
          const rhs_order = range(0, contracting_rhs.length);
          rhs_order.sort((l, r) => contracting_rhs[l] - contracting_rhs[r]);
          const contracting_indexes = rhs_order.map((i) => contracting_lhs[i]);
          const permutation = Permutation.invert(batch_lhs.concat(other_lhs, contracting_indexes));
          const transposed = this.#inner.transpose(transposed_product, permutation);
          context.addToValue(e.lhs, transposed);
          break;
        }
        case "dotGeneralRight": {
          const { contracting_lhs, contracting_rhs, batch_lhs, batch_rhs, other_lhs, other_rhs } = dotGeneralFullDims(
            e.dimensions,
            e.lhs.shape().rank(),
            value.shape().rank(),
          );

          // Compute the dimensions for the trannsposed product.
          const transposed_dims = {
            contracting_lhs: other_lhs,
            contracting_rhs: range(batch_rhs.length, other_rhs.length),
            batch_lhs,
            batch_rhs: range(0, batch_lhs.length),
          };
          const transposed_product = this.#inner.dotGeneral(e.lhs, value, transposed_dims);

          // Transposed product starts with batch dims, then lhs other dims and finally the contracting dims.
          // Let us shuffle them around. Note that the contracting dims are in the lhs order, so we first
          // need to permute the contracting dims to the index order and to the rhs order.
          const lhs_order = range(0, contracting_lhs.length);
          lhs_order.sort((l, r) => contracting_lhs[l] - contracting_lhs[r]);
          const contracting_indexes = lhs_order.map((i) => contracting_rhs[i]);
          const permutation = Permutation.invert(batch_rhs.concat(contracting_indexes, other_rhs));
          const transposed = this.#inner.transpose(transposed_product, permutation);
          context.addToValue(e.rhs, transposed);
          break;
        }
        default:
          assertNever(e);
      }
    }
    return context.inputs;
  }

  primitive(p: Primitive, inputs: GradTracer<T>[]): GradTracer<T>[] {
    switch (p.primitive) {
      case "constant":
        return [new GradTracer(this.#inner.constant(p.value), this.linearZero(p.value.shape()))];
      case "add": {
        assert.strictEqual(inputs.length, 2);
        const value = this.#inner.add(inputs[0].value, inputs[1].value);
        return [new GradTracer(value, this.linearAdd(inputs[0].grad, inputs[1].grad, value.shape()))];
      }
      case "mul": {
        assert.strictEqual(inputs.length, 2);
        const value = this.#inner.add(inputs[0].value, inputs[1].value);
        const shape = value.shape();
        const grad = this.linearAdd(
          this.linearMul(inputs[0].grad, inputs[1].value, shape),
          this.linearMul(inputs[1].grad, inputs[0].value, shape),
          shape,
        );
        return [new GradTracer(value, grad)];
      }
      case "transpose": {
        assert.strictEqual(inputs.length, 1);
        const value = this.#inner.transpose(inputs[0].value, p.permutation);
        const shape = value.shape();
        const grad = this.linearTranspose(inputs[0].grad, p.permutation, shape);
        return [new GradTracer(value, grad)];
      }
      case "broadcast": {
        assert.strictEqual(inputs.length, 1);
        const value = this.#inner.broadcast(inputs[0].value, p.new_sizes, p.broadcast_dimensions);
        const shape = value.shape();
        const grad = this.linearBroadcast(inputs[0].grad, p.new_sizes, p.broadcast_dimensions, shape);
        return [new GradTracer(value, grad)];
      }
      case "reduceSum": {
        assert.strictEqual(inputs.length, 1);
        const value = this.#inner.reduceSum(inputs[0].value, p.axes);
        const shape = value.shape();
        const grad = this.linearReduceSum(inputs[0].grad, p.axes, shape);
        return [new GradTracer(value, grad)];
      }
      case "dotGeneral": {
        assert.strictEqual(inputs.length, 2);
        const value = this.#inner.dotGeneral(inputs[0].value, inputs[1].value, p.dimensions);
        const shape = value.shape();
        const grad = this.linearAdd(
          this.linearDotGeneralLeft(inputs[0].grad, inputs[1].value, p.dimensions, shape),
          this.linearDotGeneralRight(inputs[0].value, inputs[1].grad, p.dimensions, shape),
          shape,
        );
        return [new GradTracer(value, grad)];
      }
    }
    throw new Error(`Primitive ${p} not implemented`);
  }
}

export const grad = (
  fun: <U extends Shaped>(t: Trace<U>, ...x: U[]) => U[],
  grad_input_count: number = 1,
): (<T extends Shaped>(t: Trace<T>, ...x: T[]) => T[]) => {
  return <T extends Shaped>(inner: Trace<T>, ...inputs: T[]) => {
    // Create the gradient trace and tracer for each parameter.
    const grad_input_shapes = inputs.slice(0, grad_input_count).map((v) => v.shape());
    const grad_trace = new GradTrace<T>(inner, grad_input_shapes);
    const param_tracers: GradTracer<T>[] = inputs.map((value, i) => {
      const grad: LinearExpressionValue =
        i < grad_input_count ? { kind: "input", index: i } : { kind: "zero", shape: value.shape() };
      return new GradTracer(value, grad);
    });
    // Run the function on the gradient tracer.
    const result = fun(grad_trace, ...param_tracers);
    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0].value.shape().rank(), 0, "grad only computes gradient of scalar-valued functions");
    // Evaluate the gnerated gradient graph backwards and return the result.
    return grad_trace.evaluateGraphBackwards(result[0].grad);
  };
};
