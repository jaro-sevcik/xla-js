import * as xla from "../xla-addon";
import { describe, expect, test } from "@jest/globals";

describe("XLA client", () => {
  let client: xla.Client;
  beforeEach(() => {
    client = new xla.Client();
  });

  test("can add a constant to a scalar parameter", () => {
    const builder = new xla.XlaBuilder("fn");
    const scalar_shape = xla.Shape.forArray(xla.PrimitiveType.F32, []);
    const constant_op = xla.constantR0(builder, xla.PrimitiveType.F32, 1);
    const parameter = xla.parameter(builder, 0, scalar_shape, "x");
    const computation = builder.build(xla.add(constant_op, parameter));

    // Compile the computation.
    const loaded_executable = client.compile(computation, {});

    // Execute the computation.
    const argument_literal = xla.Literal.createR0(xla.PrimitiveType.F32, 2);
    const results = loaded_executable.execute([[client.bufferFromHostLiteral(argument_literal)]], {});

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.getFirstElementF32()).toBe(3);
  });

  test("can add vectors", () => {
    const builder = new xla.XlaBuilder("fn");
    const vector2_shape = xla.Shape.forArray(xla.PrimitiveType.F32, [2]);
    const parameter1 = xla.parameter(builder, 0, vector2_shape, "x");
    const parameter2 = xla.parameter(builder, 1, vector2_shape, "y");
    const computation = builder.build(xla.add(parameter1, parameter2));

    // Compile the computation.
    const loaded_executable = client.compile(computation, {});

    // Execute the computation.
    const argument1_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [1, 2]);
    const argument2_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [3, 4]);
    const results = loaded_executable.execute(
      [[client.bufferFromHostLiteral(argument1_literal), client.bufferFromHostLiteral(argument2_literal)]],
      {}
    );

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual([4, 6]);
  });

  test("can add matrices", () => {
    const builder = new xla.XlaBuilder("fn");
    const matrix3x2_shape = xla.Shape.forArray(xla.PrimitiveType.F32, [3, 2]);
    const parameter1 = xla.parameter(builder, 0, matrix3x2_shape, "x");
    const parameter2 = xla.parameter(builder, 1, matrix3x2_shape, "y");
    const computation = builder.build(xla.add(parameter1, parameter2));

    // Compile the computation.
    const loaded_executable = client.compile(computation, {});

    // Execute the computation.
    const argument1_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [1, 2, 3, 4, 5, 6]).reshape([3, 2]);
    const argument2_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [7, 8, 9, 10, 11, 12]).reshape([3, 2]);
    const results = loaded_executable.execute(
      [[client.bufferFromHostLiteral(argument1_literal), client.bufferFromHostLiteral(argument2_literal)]],
      {}
    );

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual([8, 10, 12, 14, 16, 18]);
    expect(result_literal.shape().dimensions()).toStrictEqual([3, 2]);
  });

  test("can multiply matrices", () => {
    const builder = new xla.XlaBuilder("fn");
    const matrix2x3_shape = xla.Shape.forArray(xla.PrimitiveType.F32, [2, 3]);
    const matrix3x2_shape = xla.Shape.forArray(xla.PrimitiveType.F32, [3, 2]);
    const parameter1 = xla.parameter(builder, 0, matrix2x3_shape, "x");
    const parameter2 = xla.parameter(builder, 1, matrix3x2_shape, "y");
    const computation = builder.build(xla.dotGeneral(parameter1, parameter2, [1], [0], [], []));

    // Compile the computation.
    const loaded_executable = client.compile(computation, {});

    // Execute the computation.
    const argument1_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [1, 2, 3, 4, 5, 6]).reshape([2, 3]);
    const argument2_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [7, 8, 9, 10, 11, 12]).reshape([3, 2]);
    const results = loaded_executable.execute(
      [[client.bufferFromHostLiteral(argument1_literal), client.bufferFromHostLiteral(argument2_literal)]],
      {}
    );

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.shape().dimensions()).toStrictEqual([2, 2]);
    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual([58, 64, 139, 154]);
  });

  test("can compute sum along axis", () => {
    const add_builder = new xla.XlaBuilder("scalar_add");
    const scalar_shape = xla.Shape.forArray(xla.PrimitiveType.F32, []);
    const add_parameter1 = xla.parameter(add_builder, 0, scalar_shape, "x");
    const add_parameter2 = xla.parameter(add_builder, 1, scalar_shape, "y");
    const add_computation = add_builder.build(xla.add(add_parameter1, add_parameter2));

    const builder = new xla.XlaBuilder("fn");
    const matrix2x3_shape = xla.Shape.forArray(xla.PrimitiveType.F32, [2, 3]);
    const parameter1 = xla.parameter(builder, 0, matrix2x3_shape, "x");
    const zero = xla.constantR0(builder, xla.PrimitiveType.F32, 0);
    const computation = builder.build(xla.reduce(builder, parameter1, zero, add_computation, [1]));

    // Compile the computation.
    const loaded_executable = client.compile(computation, {});

    // Execute the computation.
    const argument1_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [1, 2, 3, 4, 5, 6]).reshape([2, 3]);
    const results = loaded_executable.execute([[client.bufferFromHostLiteral(argument1_literal)]], {});

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.shape().dimensions()).toStrictEqual([2]);
    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual([6, 15]);
  });

  test("can compute max along axis", () => {
    const max_builder = new xla.XlaBuilder("scalar_max");
    const scalar_shape = xla.Shape.forArray(xla.PrimitiveType.F32, []);
    const max_parameter1 = xla.parameter(max_builder, 0, scalar_shape, "x");
    const max_parameter2 = xla.parameter(max_builder, 1, scalar_shape, "y");
    const max_computation = max_builder.build(xla.max(max_parameter1, max_parameter2));

    const builder = new xla.XlaBuilder("fn");
    const matrix2x3_shape = xla.Shape.forArray(xla.PrimitiveType.F32, [2, 3]);
    const parameter1 = xla.parameter(builder, 0, matrix2x3_shape, "x");
    const zero = xla.constantR0(builder, xla.PrimitiveType.F32, 0);
    const computation = builder.build(xla.reduce(builder, parameter1, zero, max_computation, [0]));

    // Compile the computation.
    const loaded_executable = client.compile(computation, {});

    // Execute the computation.
    const argument1_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [1, 2, 30, 4, 5, 6]).reshape([2, 3]);
    const results = loaded_executable.execute([[client.bufferFromHostLiteral(argument1_literal)]], {});

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.shape().dimensions()).toStrictEqual([3]);
    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual([4, 5, 30]);
  });

  test("can compute max over all axes", () => {
    const max_builder = new xla.XlaBuilder("scalar_max");
    const scalar_shape = xla.Shape.forArray(xla.PrimitiveType.F32, []);
    const max_parameter1 = xla.parameter(max_builder, 0, scalar_shape, "x");
    const max_parameter2 = xla.parameter(max_builder, 1, scalar_shape, "y");
    const max_computation = max_builder.build(xla.max(max_parameter1, max_parameter2));

    const builder = new xla.XlaBuilder("fn");
    const matrix2x3_shape = xla.Shape.forArray(xla.PrimitiveType.F32, [2, 3]);
    const parameter1 = xla.parameter(builder, 0, matrix2x3_shape, "x");
    const zero = xla.constantR0(builder, xla.PrimitiveType.F32, 0);
    const computation = builder.build(xla.reduce(builder, parameter1, zero, max_computation, [0, 1]));

    // Compile the computation.
    const loaded_executable = client.compile(computation, {});

    // Execute the computation.
    const argument1_literal = xla.Literal.createR1(xla.PrimitiveType.F32, [1, 2, 30, 4, 5, 6]).reshape([2, 3]);
    const results = loaded_executable.execute([[client.bufferFromHostLiteral(argument1_literal)]], {});

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.shape().dimensions()).toStrictEqual([]);
    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual(30);
  });
});

describe("XLA literal", () => {
  test("can broadcast scalar to vector", () => {
    const shape = xla.Shape.forArray(xla.PrimitiveType.F32, [3]);
    const lit = xla.Literal.createR0(xla.PrimitiveType.F32, 1.0).broadcast(shape, []);
    expect(lit.data(xla.PrimitiveType.F32)).toStrictEqual([1, 1, 1]);
  });

  test("can get element by index", () => {
    const lit = xla.Literal.createR1(xla.PrimitiveType.F32, [1, 2]);
    expect(lit.get([1])).toStrictEqual(2);
  });
});
