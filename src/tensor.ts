import * as xla from "../xla-addon";
import { Shape } from "./shape";
import { strict as assert } from "assert";

let xlaClient = new xla.Client();

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
    return this.#ensureLiteral().data(this.#shape.element_type());
  }

  toString(): string {
    return this.#ensureLiteral().toString();
  }

  transpose(permutation?: number[]): Tensor {
    const dims = this.shape().dimensions().length;
    if (dims < 2) {
      throw new Error("Can only transpose > 2 dimensions");
    }
    if (!permutation) {
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
    return new Tensor(
      new Shape(this.shape().element_type(), new_sizes),
      results[0][0],
    );
  }

  broadcast(new_sizes: number[]): Tensor {
    const builder = new xla.XlaBuilder("reshape");
    const node = xla.parameter(builder, 0, this.#xlaShape(), "lhs");
    const computation = builder.build(xla.broadcast(node, new_sizes));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[this.#ensureBuffer()]], {});
    return new Tensor(
      new Shape(this.shape().element_type(), new_sizes),
      results[0][0],
    );
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

  #ensureLiteral(): xla.Literal {
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

  static broadcastConstant(constant: number, shape: Shape): Tensor {
    const l = xla.Literal.createR0(xla.PrimitiveType.F32, constant);
    const b = l.broadcast(shape.toXlaShape(), []);
    return new Tensor(shape, undefined, b);
  }

  static constantR0(v: number): Tensor {
    const l = xla.Literal.createR0(xla.PrimitiveType.F32, v);
    return new Tensor(new Shape(xla.PrimitiveType.F32, []), undefined, l);
  }

  static add(lhs: Tensor, rhs: Tensor) {
    assert.ok(Shape.isEqual(lhs.shape(), rhs.shape()));

    const builder = new xla.XlaBuilder("add");
    const lhs_node = xla.parameter(builder, 0, lhs.#xlaShape(), "lhs");
    const rhs_node = xla.parameter(builder, 1, lhs.#xlaShape(), "rhs");
    const computation = builder.build(xla.add(lhs_node, rhs_node));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute(
      [[lhs.#ensureBuffer(), rhs.#ensureBuffer()]],
      {},
    );

    return new Tensor(lhs.shape(), results[0][0]);
  }

  static mul(lhs: Tensor, rhs: Tensor) {
    assert.ok(Shape.isEqual(lhs.shape(), rhs.shape()));

    const builder = new xla.XlaBuilder("mul");
    const lhs_node = xla.parameter(builder, 0, lhs.#xlaShape(), "lhs");
    const rhs_node = xla.parameter(builder, 1, lhs.#xlaShape(), "rhs");
    const computation = builder.build(xla.mul(lhs_node, rhs_node));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute(
      [[lhs.#ensureBuffer(), rhs.#ensureBuffer()]],
      {},
    );

    return new Tensor(lhs.shape(), results[0][0]);
  }

  static dotGeneral(
    lhs: Tensor,
    rhs: Tensor,
    contracting_lhs: number[],
    contracting_rhs: number[],
    batch_lhs: number[],
    batch_rhs: number[],
  ): Tensor {
    const shape = Shape.dotGeneral(
      lhs.shape(),
      rhs.shape(),
      contracting_lhs,
      contracting_rhs,
      batch_lhs,
      batch_rhs,
    );

    const builder = new xla.XlaBuilder("dotGeneral");
    const lhs_node = xla.parameter(builder, 0, lhs.#xlaShape(), "lhs");
    const rhs_node = xla.parameter(builder, 1, lhs.#xlaShape(), "rhs");
    const computation = builder.build(
      xla.dotGeneral(
        lhs_node,
        rhs_node,
        contracting_lhs,
        contracting_rhs,
        batch_lhs,
        batch_rhs,
      ),
    );
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute(
      [[lhs.#ensureBuffer(), rhs.#ensureBuffer()]],
      {},
    );

    return new Tensor(shape, results[0][0]);
  }

  static matmul(lhs: Tensor, rhs: Tensor) {
    const dims = lhs.shape().rank();
    const batch = new Array(dims).fill(0).map((e, i) => i);
    return Tensor.dotGeneral(lhs, rhs, [dims - 1], [dims - 2], batch, batch);
  }

  static arange(shape: Shape, dimension: number) {
    const builder = new xla.XlaBuilder("iota");
    const computation = builder.build(xla.iota(builder, shape, dimension));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[]], {});

    return new Tensor(shape, results[0][0]);
  }
}
