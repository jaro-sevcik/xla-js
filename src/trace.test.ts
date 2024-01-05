import { Tensor } from "./tensor";
import { EvalTrace, Shaped, Trace, grad } from "./trace";

function testfn<T extends Shaped>(trace: Trace<T>, x: T): T[] {
  const exprX2 = trace.mul(x, x);
  const expr4X = trace.mul(trace.constant(Tensor.constantR0(4)), x);
  return [trace.add(exprX2, trace.add(expr4X, trace.constant(Tensor.constantR0(6))))];
}

describe("Trace", () => {
  const trace = new EvalTrace();

  test("can evaluate a scalar mul-add expression", () => {
    expect(testfn(trace, Tensor.literal(3))[0].data()).toStrictEqual([27]);
  });

  test("can compute gradient of a scalar expression", () => {
    const grad_testfn = grad(testfn);
    expect(grad_testfn(trace, Tensor.literal(3))[0].data()).toStrictEqual([10]);
  });

  test("can compute second gradient of a scalar expression", () => {
    const grad_testfn = grad(grad(testfn));
    expect(grad_testfn(trace, Tensor.literal(3))[0].data()).toStrictEqual([2]);
  });

  test("can evaluate vector-add expression", () => {
    const vec2 = Tensor.literal([3, 4]);

    function add<T extends Shaped>(t: Trace<T>, x: T): T[] {
      return [t.add(t.literal([1, 2]), x)];
    }

    expect(add(trace, vec2)[0].toLiteral()).toStrictEqual([4, 6]);
  });

  test("can evaluate matmul", () => {
    function mm<T extends Shaped>(t: Trace<T>, x: T): T[] {
      return [
        t.matmul(
          t.literal([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          x
        ),
      ];
    }

    const x = Tensor.literal([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    expect(mm(trace, x)[0].toLiteral()).toStrictEqual([
      [22, 28],
      [49, 64],
    ]);
  });

  test("can compute sum via dot-product", () => {
    function sum<T extends Shaped>(t: Trace<T>, x: T): T[] {
      const ones = t.constant(Tensor.ones(x.shape()));
      return [t.dotGeneral(x, ones, { contracting_lhs: [0], contracting_rhs: [0], batch_lhs: [], batch_rhs: [] })];
    }

    const x = Tensor.literal([1, 2, 3, 4]);
    expect(sum(trace, x)[0].toLiteral()).toStrictEqual(10);
  });

  test("can compute gradient of sum via dot-product", () => {
    function sum<T extends Shaped>(t: Trace<T>, x: T): T[] {
      const ones = t.constant(Tensor.ones(x.shape()));
      return [t.dotGeneral(x, ones, { contracting_lhs: [0], contracting_rhs: [0], batch_lhs: [], batch_rhs: [] })];
    }

    const grad_sum = grad(sum);

    const x = Tensor.literal([1, 2, 3, 4]);
    expect(grad_sum(trace, x)[0].toLiteral()).toStrictEqual([1, 1, 1, 1]);
  });

  test("can compute gradient of 2d sum via dot-product", () => {
    function sum<T extends Shaped>(t: Trace<T>, x: T): T[] {
      const ones = t.constant(Tensor.ones(x.shape()));
      return [t.dotGeneral(x, ones, { contracting_lhs: [0, 1], contracting_rhs: [0, 1], batch_lhs: [], batch_rhs: [] })];
    }

    const grad_sum = grad(sum);

    const x = Tensor.literal([[1, 2], [3, 4]]);
    expect(grad_sum(trace, x)[0].toLiteral()).toStrictEqual([[1, 1], [1, 1]]);
  });

  test("can compute gradient of 2d square dot-product", () => {
    function sum<T extends Shaped>(t: Trace<T>, x: T): T[] {
      return [t.dotGeneral(x, x, { contracting_lhs: [0, 1], contracting_rhs: [0, 1], batch_lhs: [], batch_rhs: [] })];
    }

    const grad_sum = grad(sum);

    const x = Tensor.literal([[1, 2], [3, 4]]);
    expect(grad_sum(trace, x)[0].toLiteral()).toStrictEqual([[2, 4], [6, 8]]);
  });

  test("can compute gradient of 2d square dot-product swapped rhs contraction", () => {
    function sum<T extends Shaped>(t: Trace<T>, x: T): T[] {
      return [t.dotGeneral(x, x, { contracting_lhs: [0, 1], contracting_rhs: [1, 0], batch_lhs: [], batch_rhs: [] })];
    }
    const grad_sum = grad(sum);
    const x = Tensor.literal([[1, 2], [3, 4]]);
    expect(grad_sum(trace, x)[0].toLiteral()).toStrictEqual([[2, 6], [4, 8]]);
  });

  test("can compute gradient of 2d square dot-product swapped lhs contraction", () => {
    function sum<T extends Shaped>(t: Trace<T>, x: T): T[] {
      return [t.dotGeneral(x, x, { contracting_lhs: [1, 0], contracting_rhs: [0, 1], batch_lhs: [], batch_rhs: [] })];
    }
    const grad_sum = grad(sum);
    const x = Tensor.literal([[1, 2], [3, 4]]);
    expect(grad_sum(trace, x)[0].toLiteral()).toStrictEqual([[2, 6], [4, 8]]);
  });

  test("can compute gradient of transpose (and dot-product)", () => {
    function sum<T extends Shaped>(t: Trace<T>, x: T): T[] {
      const transposed = t.transpose(x, [1, 0]);
      return [t.dotGeneral(x, transposed, { contracting_lhs: [0, 1], contracting_rhs: [0, 1], batch_lhs: [], batch_rhs: [] })];
    }
    const grad_sum = grad(sum);
    const x = Tensor.literal([[1, 2], [3, 4]]);
    expect(grad_sum(trace, x)[0].toLiteral()).toStrictEqual([[2, 6], [4, 8]]);
  });  
});
