import * as xla from "../xla-addon";
import { Shape, Shaped } from "./shape";
import { assertNever, output_shapes, Primitive, Trace } from "./trace";
import { strict as assert } from "assert";

export class ShapedOp implements Shaped {
  #shape: Shape;
  #op: xla.XlaOp;

  constructor(shape: Shape, op: xla.XlaOp) {
    this.#shape = shape;
    this.#op = op;
  }

  shape() {
    return this.#shape;
  }

  op() {
    return this.#op;
  }
}

export class ShapeTrace extends Trace<Shape> {
  primitive(p: Primitive, inputs: Shape[]): Shape[] {
    return output_shapes(p, inputs);
  }
}

export class XlaTrace extends Trace<ShapedOp> {
  #builder: xla.XlaBuilder;

  constructor(name: string) {
    super();
    this.#builder = new xla.XlaBuilder(name);
  }

  primitive(p: Primitive, inputs: ShapedOp[]): ShapedOp[] {
    const shapes = output_shapes(
      p,
      inputs.map((o) => o.shape()),
    );
    switch (p.primitive) {
      case "constant": {
        assert.strictEqual(inputs.length, 0);
        const op = xla.constantLiteral(this.#builder, p.value.ensureXlaLiteral());
        return [new ShapedOp(shapes[0], op)];
      }
      case "add":
        assert.strictEqual(inputs.length, 2);
        return [new ShapedOp(shapes[0], xla.add(inputs[0].op(), inputs[1].op()))];
      case "mul":
        assert.strictEqual(inputs.length, 2);
        return [new ShapedOp(shapes[0], xla.mul(inputs[0].op(), inputs[1].op()))];
      case "dotGeneral": {
        assert.strictEqual(inputs.length, 2);
        return [
          new ShapedOp(
            shapes[0],
            xla.dotGeneral(
              inputs[0].op(),
              inputs[1].op(),
              p.dimensions.contracting_lhs,
              p.dimensions.contracting_rhs,
              p.dimensions.batch_lhs,
              p.dimensions.batch_rhs,
            ),
          ),
        ];
      }
      case "transpose":
        assert.strictEqual(inputs.length, 1);
        return [new ShapedOp(shapes[0], xla.transpose(inputs[0].op(), p.permutation))];
      case "reshape":
        assert.strictEqual(inputs.length, 1);
        return [new ShapedOp(shapes[0], xla.reshape(inputs[0].op(), p.new_sizes))];
      case "broadcast":
        assert.strictEqual(inputs.length, 1);
        return [new ShapedOp(shapes[0], xla.broadcastInDim(inputs[0].op(), p.new_sizes, p.broadcast_dimensions))];
      case "reduceSum":
        assert.strictEqual(inputs.length, 1);

        // TODO(jarin) Cache the scalar adder!
        const add_builder = new xla.XlaBuilder("scalar_add");
        const scalar_shape = xla.Shape.forArray(xla.PrimitiveType.F32, []);
        const add_parameter1 = xla.parameter(add_builder, 0, scalar_shape, "x");
        const add_parameter2 = xla.parameter(add_builder, 1, scalar_shape, "y");
        const add_computation = add_builder.build(xla.add(add_parameter1, add_parameter2));
        const zero = xla.constantR0(this.#builder, inputs[0].shape().element_type(), 0);

        return [new ShapedOp(shapes[0], xla.reduce(this.#builder, inputs[0].op(), zero, add_computation, p.axes))];
      case "block":
        throw new Error(`Block not implemented`);
    }
    assertNever(p);
  }
}
