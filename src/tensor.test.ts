import { Tensor } from "./tensor";

describe("Tensor", () => {
  test("can add scalars", () => {
    const lhs = Tensor.constantR0(1);
    const rhs = Tensor.constantR0(2);
    const res = Tensor.add(lhs, rhs).data()[0];
    expect(res).toBe(3);
  });
});

describe("Tensor", () => {
  test("can multiply scalars", () => {
    const lhs = Tensor.constantR0(2);
    const rhs = Tensor.constantR0(3);
    const res = Tensor.mul(lhs, rhs).data()[0];
    expect(res).toBe(6);
  });

  test("stringifies correctly", () => {
    expect(Tensor.constantR0(3).toString()).toBe('f32[] 3');
  });

  test("matrix literal and back", () => {
    const l = [[1, 2, 3], [4, 5, 6]];
    expect(Tensor.literal(l).toLiteral()).toStrictEqual(l);
  });

  test("scalar literal and back", () => {
    const l = 42;
    expect(Tensor.literal(l).toLiteral()).toStrictEqual(l);
  });
});
