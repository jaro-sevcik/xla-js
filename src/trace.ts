import { Shape } from "./shape";
import { Tensor } from "./tensor";
import { strict as assert } from 'assert';

interface Shaped {
  shape(): Shape;
}

export type Mul = {
  primitive: "mul";
};

export type Add = {
  primitive: "add";
};

export type DotGeneral = {
  primitive: "dot_general";
  batch_lhs: number[];
  batch_rhs: number[];
  contracting_lhs: number[];
  contracting_rhs: number[];
};

export type Transpose = {
  primitive: "transpose";
  permutation: number[];
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
  | DotGeneral
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
      assert.strictEqual(input_shapes.length, 2);
      assert.ok(Shape.isEqual(input_shapes[0], input_shapes[1]));
      return [input_shapes[0]];
    case "mul":
      assert.strictEqual(input_shapes.length, 2);
      assert.ok(Shape.isEqual(input_shapes[0], input_shapes[1]));
      return [input_shapes[0]];
    case "dot_general":
      assert.strictEqual(input_shapes.length, 2);
      return [Shape.dotGeneral(input_shapes[0], input_shapes[1], this.batch_lhs, this.batch_rhs, this.contracting_lhs, this.contracting_rhs)];
    case "transpose":
      assert.strictEqual(input_shapes.length, 1);
      return [input_shapes[0].transpose(this.permutation)];
    case "block":
      // TODO
      return [];
    case "reshape":
      assert.strictEqual(input_shapes.length, 1);
      assert.strictEqual(input_shapes[0].total_size(), this.shape.total_size());
      return [this.shape];
    case "broadcast":
      assert.strictEqual(input_shapes.length, 1);
      assert.strictEqual(input_shapes[0].total_size(), this.shape.total_size());
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
        assert.strictEqual(inputs.length, 2);
        return [Tensor.add(inputs[0], inputs[1])];
      case "mul":
        assert.strictEqual(inputs.length, 2);
        return [Tensor.mul(inputs[0], inputs[1])];
    }

    throw new Error(`Primitive ${p} not implemented`);
  }
}

type LinearExpressionInput = { kind: "input"; index: number };
type LinearExpressionIndex = { kind: "expr"; index: number };
type LinearExpressionZero = { kind: "zero"; shape: Shape };
type LinearExpressionValue =
  | LinearExpressionInput
  | LinearExpressionIndex
  | LinearExpressionZero;

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
type LinearExpression<T> = LinearExpressionMul<T> | LinearExpressionAdd;

class LinearGraph<T> {
  expressions: { shape: Shape; expression: LinearExpression<T> }[] = [];

  constructor(readonly input_shapes: Shape[]) {}
}

class LinearExpressionEvakuationContext<T> {
  values: T[];
  inputs: T[];
  #trace: Trace<T>;

  constructor(
    input_shapes: Shape[],
    expression_shapes: Shape[],
    trace: Trace<T>,
  ) {
    this.values = expression_shapes.map((s) => trace.constant(Tensor.zeros(s)));
    this.inputs = input_shapes.map((s) => trace.constant(Tensor.zeros(s)));
    this.#trace = trace;
  }

  addToValue(index: LinearExpressionValue, value: T) {
    switch (index.kind) {
      case "expr":
        this.values[index.index] = this.#trace.add(
          this.values[index.index],
          value,
        );
        break;
      case "input":
        this.inputs[index.index] = this.#trace.add(
          this.inputs[index.index],
          value,
        );
        break;
      case "zero":
        break;
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

  linearAdd(
    lhs: LinearExpressionValue,
    rhs: LinearExpressionValue,
    shape: Shape,
  ): LinearExpressionValue {
    if (lhs.kind === "zero") return rhs;
    if (rhs.kind === "zero") return lhs;
    const index = this.addExpression(shape, { kind: "add", lhs, rhs });
    return { kind: "expr", index };
  }

  linearMul(
    lhs: LinearExpressionValue,
    rhs: T,
    shape: Shape,
  ): LinearExpressionValue {
    if (lhs.kind === "zero") return lhs;
    const index = this.addExpression(shape, { kind: "mul", lhs, rhs });
    return { kind: "expr", index };
  }

  linearZero(shape: Shape): LinearExpressionValue {
    return { kind: "zero", shape };
  }

  evaluateGraphBackwards(from: LinearExpressionValue): T[] {
    const expressions = this.#linearGraph.expressions;
    const expression_shapes = expressions.map(({ shape }) => shape);
    const context = new LinearExpressionEvakuationContext(
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
        default:
          assertNever(e);
      }
    }
    return context.inputs;
  }

  primitive(p: Primitive, inputs: GradTracer<T>[]): GradTracer<T>[] {
    switch (p.primitive) {
      case "constant":
        return [
          new GradTracer(
            this.#inner.constant(p.value),
            this.linearZero(p.value.shape()),
          ),
        ];
      case "add": {
        assert.strictEqual(inputs.length, 2);
        const value = this.#inner.add(inputs[0].value, inputs[1].value);
        return [
          new GradTracer(
            value,
            this.linearAdd(inputs[0].grad, inputs[1].grad, value.shape()),
          ),
        ];
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
    const grad_input_shapes = inputs
      .slice(0, grad_input_count)
      .map((v) => v.shape());
    const grad_trace = new GradTrace<T>(inner, grad_input_shapes);
    const param_tracers: GradTracer<T>[] = inputs.map((value, i) => {
      const grad: LinearExpressionValue =
        i < grad_input_count
          ? { kind: "input", index: i }
          : { kind: "zero", shape: value.shape() };
      return new GradTracer(value, grad);
    });
    // Run the function on the gradient tracer.
    const result = fun(grad_trace, ...param_tracers);
    assert.strictEqual(result.length, 1);
    // Evaluate the gnerated gradient graph backwards and return the result.
    return grad_trace.evaluateGraphBackwards(result[0].grad);
  };
};
