
import {Tensor} from './tensor';
import {EvalTrace, Trace} from './trace';

function testfn<T>(trace: Trace<T>, x: T) {
  const exprX2 = trace.mul(x, x);
  const expr4X = trace.mul(trace.constant(Tensor.constantR0(4)), x);
  return trace.add(
    exprX2,
    trace.add(expr4X, trace.constant(Tensor.constantR0(6))),
  );
}

describe("Trace", () => {
  test("can evaluate scale mul-add expression", () => {
    const trace = new EvalTrace();

    expect(testfn(trace, Tensor.constantR0(3.0)).data()).toStrictEqual([27]);
  });
});
