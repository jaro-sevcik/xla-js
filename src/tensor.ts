import * as xla from "../xla-addon";
import { DotGeneralDimensions, PrimitiveType, Shape } from "./shape";
import { strict as assert } from "assert";

let xlaClient = new xla.Client();

export type TensorLiteral = number | TensorLiteral[];

export class Tensor {
  #buffer?: xla.PjRtBuffer;
  #literal?: xla.Literal;
  readonly #shape: Shape;

  constructor(shape: Shape, buffer?: xla.PjRtBuffer, literal?: xla.Literal) {
    this.#buffer = buffer;
    this.#literal = literal;
    this.#shape = shape;
  }

  shape(): Shape {
    return this.#shape;
  }

  data(): number[] {
    return this.ensureXlaLiteral().data(this.#shape.element_type());
  }

  toString(): string {
    return this.ensureXlaLiteral().toString();
  }

  transpose(permutation?: number[]): Tensor {
    const dims = this.shape().dimensions().length;
    if (!permutation) {
      if (dims < 2) {
        throw new Error("Can only transpose > 2 dimensions");
      }
      permutation = [];
      for (let i = 0; i < dims - 2; i++) permutation.push(i);
      permutation.push(dims - 1, dims - 2);
    }
    const builder = new xla.XlaBuilder("transpose");
    const node = xla.parameter(builder, 0, this.#xlaShape(), "lhs");
    const computation = builder.build(xla.transpose(node, permutation));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[this.#ensureBuffer()]], {});
    return new Tensor(this.shape().transpose(permutation), results[0][0]);
  }

  reshape(new_sizes: number[]): Tensor {
    const builder = new xla.XlaBuilder("reshape");
    const node = xla.parameter(builder, 0, this.#xlaShape(), "lhs");
    const computation = builder.build(xla.reshape(node, new_sizes));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[this.#ensureBuffer()]], {});
    return new Tensor(new Shape(this.shape().element_type(), new_sizes), results[0][0]);
  }

  broadcastInDim(new_sizes: number[], broadcast_dimensions: number[]): Tensor {
    const builder = new xla.XlaBuilder("reshape");
    const node = xla.parameter(builder, 0, this.#xlaShape(), "lhs");
    const computation = builder.build(xla.broadcastInDim(node, new_sizes, broadcast_dimensions));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[this.#ensureBuffer()]], {});
    return new Tensor(new Shape(this.shape().element_type(), new_sizes), results[0][0]);
  }

  reduceSum(axes: number[]): Tensor {
    // TODO(jarin) Cache the scalar adder!
    const add_builder = new xla.XlaBuilder("scalar_add");
    const scalar_shape = xla.Shape.forArray(xla.PrimitiveType.F32, []);
    const add_parameter1 = xla.parameter(add_builder, 0, scalar_shape, "x");
    const add_parameter2 = xla.parameter(add_builder, 1, scalar_shape, "y");
    const add_computation = add_builder.build(xla.add(add_parameter1, add_parameter2));

    const builder = new xla.XlaBuilder("reduceSum");
    const node = xla.parameter(builder, 0, this.#xlaShape(), "input");
    const zero = xla.constantR0(builder, this.shape().element_type(), 0);
    const computation = builder.build(xla.reduce(builder, node, zero, add_computation, axes));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[this.#ensureBuffer()]], {});
    return new Tensor(this.shape().removeAxes(axes), results[0][0]);
  }

  toLiteral(): TensorLiteral {
    const rank = this.shape().rank();
    let data = this.data();
    if (rank === 0) return data[0];

    const dims = this.shape().dimensions();
    const stack: TensorLiteral[][] = [];

    for (let i = 0; i < rank; i++) {
      stack[i] = [];
      if (i > 0) {
        stack[i - 1].push(stack[i]);
      }
    }

    for (let i = 0; i < data.length; i++) {
      let e: TensorLiteral = data[i];

      for (let d = rank - 1; d >= 0; d--) {
        if (stack[d].length < dims[d]) {
          stack[d].push(e);
          break;
        }
        stack[d] = [e];
        e = stack[d];
      }
    }
    return stack[0];
  }

  #ensureBuffer(): xla.PjRtBuffer {
    if (!this.#buffer) {
      if (!this.#literal) {
        throw new Error("Internal error: Missing literal or buffer on tensor");
      }
      this.#buffer = xlaClient.bufferFromHostLiteral(this.#literal);
    }
    return this.#buffer;
  }

  ensureXlaLiteral(): xla.Literal {
    if (!this.#literal) {
      if (!this.#buffer) {
        throw new Error("Internal error: Missing literal or buffer on tensor");
      }
      this.#literal = this.#buffer.toLiteralSync();
    }
    return this.#literal;
  }

  #xlaShape(): xla.Shape {
    if (this.#literal) return this.#literal.shape();

    return this.#shape.toXlaShape();
  }

  static zeros(shape: Shape): Tensor {
    return Tensor.broadcastConstant(0, shape);
  }

  static ones(shape: Shape): Tensor {
    return Tensor.broadcastConstant(1, shape);
  }

  static broadcastConstant(constant: number, shape: Shape): Tensor {
    const l = xla.Literal.createR0(shape.element_type(), constant);
    const b = l.broadcast(shape.toXlaShape(), []);
    return new Tensor(shape, undefined, b);
  }

  static constantR0(v: number): Tensor {
    const l = xla.Literal.createR0(xla.PrimitiveType.F32, v);
    return new Tensor(new Shape(xla.PrimitiveType.F32, []), undefined, l);
  }

  static constantR1(v: number[]): Tensor {
    const l = xla.Literal.createR1(xla.PrimitiveType.F32, v);
    return new Tensor(new Shape(xla.PrimitiveType.F32, [v.length]), undefined, l);
  }

  static add(lhs: Tensor, rhs: Tensor) {
    assert.ok(Shape.isEqual(lhs.shape(), rhs.shape()));

    const builder = new xla.XlaBuilder("add");
    const lhs_node = xla.parameter(builder, 0, lhs.#xlaShape(), "lhs");
    const rhs_node = xla.parameter(builder, 1, lhs.#xlaShape(), "rhs");
    const computation = builder.build(xla.add(lhs_node, rhs_node));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[lhs.#ensureBuffer(), rhs.#ensureBuffer()]], {});

    return new Tensor(lhs.shape(), results[0][0]);
  }

  static mul(lhs: Tensor, rhs: Tensor) {
    assert.ok(Shape.isEqual(lhs.shape(), rhs.shape()));

    const builder = new xla.XlaBuilder("mul");
    const lhs_node = xla.parameter(builder, 0, lhs.#xlaShape(), "lhs");
    const rhs_node = xla.parameter(builder, 1, lhs.#xlaShape(), "rhs");
    const computation = builder.build(xla.mul(lhs_node, rhs_node));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[lhs.#ensureBuffer(), rhs.#ensureBuffer()]], {});

    return new Tensor(lhs.shape(), results[0][0]);
  }

  static dotGeneral(lhs: Tensor, rhs: Tensor, dotDims: DotGeneralDimensions): Tensor {
    const shape = Shape.dotGeneral(lhs.shape(), rhs.shape(), dotDims);

    const builder = new xla.XlaBuilder("dotGeneral");
    const lhs_node = xla.parameter(builder, 0, lhs.#xlaShape(), "lhs");
    const rhs_node = xla.parameter(builder, 1, rhs.#xlaShape(), "rhs");

    const computation = builder.build(
      xla.dotGeneral(
        lhs_node,
        rhs_node,
        dotDims.contracting_lhs,
        dotDims.contracting_rhs,
        dotDims.batch_lhs,
        dotDims.batch_rhs
      )
    );
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[lhs.#ensureBuffer(), rhs.#ensureBuffer()]], {});

    return new Tensor(shape, results[0][0]);
  }

  static matmul(lhs: Tensor, rhs: Tensor) {
    const rank = lhs.shape().rank();
    const batch = new Array(rank).fill(0).map((e, i) => i);
    return Tensor.dotGeneral(lhs, rhs, {
      contracting_lhs: [rank - 1],
      contracting_rhs: [rank - 2],
      batch_lhs: batch,
      batch_rhs: batch,
    });
  }

  static arange(shape: Shape, dimension: number) {
    const builder = new xla.XlaBuilder("iota");
    const computation = builder.build(xla.iota(builder, shape, dimension));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[]], {});

    return new Tensor(shape, results[0][0]);
  }

  static literal(t: TensorLiteral, p: PrimitiveType = PrimitiveType.F32): Tensor {
    if (typeof t === "number") {
      return Tensor.constantR0(t);
    }

    const flat: number[] = [];
    const dimensions = [];
    const index = [];
    const array: TensorLiteral[][] = [];
    let current: TensorLiteral = t;

    // Figure out the dimensions.
    while (typeof current !== "number") {
      assert.ok(current.length > 0, "Literal arrays must be non-empty");
      dimensions.push(current.length);
      index.push(0);
      array.push(current);
      current = current[0];
    }

    const rank = dimensions.length;
    // Flatten the array while doing some sanity checks on the shape.
    while (true) {
      const element = array[rank - 1][index[rank - 1]];
      assert.ok(typeof element === "number", "Ragged constants are not supported");
      flat.push(element);

      let d;
      for (d = rank - 1; d >= 0; d--) {
        index[d]++;
        if (index[d] < dimensions[d]) break;
      }
      if (d < 0) break;
      for (d = d + 1; d < rank; d++) {
        index[d] = 0;
        const next: number | TensorLiteral[] = array[d - 1][index[d - 1]];
        if (typeof next === "number") throw new Error("Ragged constants are not supported");
        array[d] = next;
      }
    }

    return Tensor.constantR1(flat).reshape(dimensions);
  }
}
