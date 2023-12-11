import * as xla from "../xla-addon";
import { Shape } from "./shape";

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

  shape(): Shape { return this.#shape; }

  data(): number[] {
    return this.#ensureLiteral().data(this.#shape.element_type());
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

  static constantR0(v: number): Tensor {
    const l = xla.Literal.createR0(xla.PrimitiveType.F32, v);
    return new Tensor(new Shape(xla.PrimitiveType.F32, []), undefined, l);
  }

  static add(lhs: Tensor, rhs: Tensor) {
    console.assert(Shape.isEqual(lhs.shape(), rhs.shape()));

    const builder = new xla.XlaBuilder("add");
    const lhs_node = xla.parameter(builder, 0, lhs.#xlaShape(), "lhs");
    const rhs_node = xla.parameter(builder, 1, lhs.#xlaShape(), "rhs");
    const computation = builder.build(xla.add(lhs_node, rhs_node));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[lhs.#ensureBuffer(), rhs.#ensureBuffer()]], {});


    return new Tensor(lhs.shape(), results[0][0]);
  }

  static mul(lhs: Tensor, rhs: Tensor) {
    console.assert(Shape.isEqual(lhs.shape(), rhs.shape()));

    const builder = new xla.XlaBuilder("mul");
    const lhs_node = xla.parameter(builder, 0, lhs.#xlaShape(), "lhs");
    const rhs_node = xla.parameter(builder, 1, lhs.#xlaShape(), "rhs");
    const computation = builder.build(xla.mul(lhs_node, rhs_node));
    const executable = xlaClient.compile(computation, {});
    const results = executable.execute([[lhs.#ensureBuffer(), rhs.#ensureBuffer()]], {});

    return new Tensor(lhs.shape(), results[0][0]);
  }
}
