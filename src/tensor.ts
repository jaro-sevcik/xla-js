import * as xla from "../xla-addon";
import { Shape } from "./shape";

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

  static constantR0(v: number): Tensor {
    // TODO value literal!
    return new Tensor(new Shape(xla.PrimitiveType.F32, []));
  }
}
