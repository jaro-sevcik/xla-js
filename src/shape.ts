export enum PrimitiveType {
  F32 = "F32",
}

export class Shape {
  #dimensions: number[];
  #type: PrimitiveType;

  constructor(type: PrimitiveType, dimensions: number[]) {
    this.#dimensions = dimensions;
    this.#type = type;
  }

  total_size(): number {
    return this.#dimensions.reduce((s, d) => s*d, 1);
  }


  static matmul(lhs: Shape, rhs: Shape): Shape {
    console.assert(lhs.#dimensions.length === rhs.#dimensions.length);
    console.assert(lhs.#dimensions.length >= 2);
    console.assert(Shape.isEqualUpto(lhs, rhs, lhs.#dimensions.length - 2));
    const dimensions = [...lhs.#dimensions].splice(
      lhs.#dimensions.length - 1,
      1,
      rhs.#dimensions[rhs.#dimensions.length - 1],
    );
    return new Shape(lhs.#type, dimensions);
  }

  static transpose(input: Shape): Shape {
    console.assert(input.#dimensions.length >= 2);
    const dimensions = [...input.#dimensions].splice(
      input.#dimensions.length - 2,
      2,
      input.#dimensions[input.#dimensions.length - 1],
      input.#dimensions[input.#dimensions.length - 2]
    );
    return new Shape(input.#type, dimensions);
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
