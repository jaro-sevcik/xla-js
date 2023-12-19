import { Tensor } from "./tensor";
import { EvalTrace, Trace, grad } from "./trace";

function testfn<T>(trace: Trace<T>, x: T): T[] {
  const exprX2 = trace.mul(x, x);
  const expr4X = trace.mul(trace.constant(Tensor.constantR0(4)), x);
  return [
    trace.add(exprX2, trace.add(expr4X, trace.constant(Tensor.constantR0(6)))),
  ];
}

describe("Trace", () => {
  const trace = new EvalTrace();
  const const3 = Tensor.constantR0(3.0);
  const vec2 = Tensor.literal([3, 4]);

  test("can evaluate a scalar mul-add expression", () => {
    expect(testfn(trace, const3)[0].data()).toStrictEqual([27]);
  });

  test("can compute gradient of a scalar expression", () => {
    const grad_testfn = grad(testfn);
    expect(grad_testfn(trace, const3)[0].data()).toStrictEqual([10]);
  });

  test("can compute second gradient of a scalar expression", () => {
    const grad_testfn = grad(grad(testfn));
    expect(grad_testfn(trace, const3)[0].data()).toStrictEqual([2]);
  });

  test("can evaluate vector-add expression", () => {
    function add<T>(t: Trace<T>, x: T): T[] {
      return [t.add(t.constant(Tensor.literal([1, 2])), x)];
    }

    expect(add(trace, vec2)[0].toLiteral()).toStrictEqual([4, 6]);
  });
});
