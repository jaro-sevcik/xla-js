import { Tensor } from './tensor';

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
});
