import * as xla from "../xla-addon";
import { strict as assert } from "assert";

export type PrimitiveType = xla.PrimitiveType;
export const PrimitiveType = xla.PrimitiveType;

export type DotGeneralDimensions = {
  contracting_lhs: number[];
  contracting_rhs: number[];
  batch_lhs: number[];
  batch_rhs: number[];
};

export class Shape {
  #dimensions: number[];
  #type: PrimitiveType;

  constructor(type: PrimitiveType, dimensions: number[]) {
    this.#dimensions = dimensions;
    this.#type = type;
  }

  total_size(): number {
    return this.#dimensions.reduce((s, d) => s * d, 1);
  }

  element_type(): PrimitiveType {
    return this.#type;
  }

  dimensions(): number[] {
    return this.#dimensions;
  }

  rank(): number {
    return this.#dimensions.length;
  }

  toXlaShape(): xla.Shape {
    return xla.Shape.forArray(this.#type, this.#dimensions);
  }

  static dotGeneral(
    lhs: Shape,
    rhs: Shape,
    { contracting_lhs, contracting_rhs, batch_lhs, batch_rhs }: DotGeneralDimensions,
  ): Shape {
    const lhs_dims = [...lhs.dimensions()];
    const rhs_dims = [...rhs.dimensions()];
    const result = [];

    assert.strictEqual(batch_lhs.length, batch_rhs.length, "Numbers of batch dimensions must match");
    assert.strictEqual(lhs.element_type(), rhs.element_type(), "Element types must match");

    for (let i = 0; i < batch_lhs.length; i++) {
      assert.strictEqual(lhs_dims[batch_lhs[i]], rhs_dims[batch_rhs[i]], "Batch dimension sizes must match");
      result.push(lhs_dims[batch_lhs[i]]);
      lhs_dims[batch_lhs[i]] = -1;
      rhs_dims[batch_rhs[i]] = -1;
    }

    assert.strictEqual(batch_lhs.length, batch_rhs.length, "Number of contracting dimensions must match");
    for (let i = 0; i < contracting_lhs.length; i++) {
      assert.strictEqual(
        lhs_dims[contracting_lhs[i]],
        rhs_dims[contracting_rhs[i]],
        "Contracting dimension sizes must match",
      );
      lhs_dims[contracting_lhs[i]] = -1;
      rhs_dims[contracting_rhs[i]] = -1;
    }

    for (const d of lhs_dims) {
      if (d >= 0) result.push(d);
    }
    for (const d of rhs_dims) {
      if (d >= 0) result.push(d);
    }

    return new Shape(lhs.#type, result);
  }

  transpose(permutation: number[]): Shape {
    const dimensions = Array(this.#dimensions.length).fill(-1);
    assert.strictEqual(permutation.length, dimensions.length);
    for (let i = 0; i < permutation.length; i++) {
      assert.ok(dimensions[i] === -1, "Duplicate indices in transpose permutation");
      dimensions[i] = this.#dimensions[permutation[i]];
    }
    return new Shape(this.#type, dimensions);
  }

  removeAxes(axes: number[]): Shape {
    const rank = this.rank();
    const dimensions = [...this.dimensions()];
    let last = -1;
    let removed = 0;
    for (const axis of axes) {
      assert.ok(last < axis, "Axes must be in ascending order");
      assert.ok(axis < rank, "Axes must be lower than rank");
      dimensions.splice(axis - removed, 1);
      removed++;
    }
    return new Shape(this.#type, dimensions);
  }

  static isEqual(lhs: Shape, rhs: Shape): boolean {
    if (lhs.#dimensions.length !== rhs.#dimensions.length) {
      return false;
    }
    if (lhs.#type !== rhs.#type) return false;
    return Shape.isEqualUpto(lhs, rhs, lhs.#dimensions.length);
  }

  static isEqualUpto(lhs: Shape, rhs: Shape, upto: number) {
    if (lhs.#type !== rhs.#type) return false;
    for (let i = 0; i < upto; i++) {
      if (lhs.#dimensions[i] !== rhs.#dimensions[i]) {
        return false;
      }
    }
    return true;
  }
}
