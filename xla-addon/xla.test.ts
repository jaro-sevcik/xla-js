import * as xla from '../xla-addon';
import {describe, expect, test} from '@jest/globals';

describe('XLA client', () => {
  let client: xla.Client;
  beforeEach(() => {
    client = new xla.Client();
  });

  test('can add a constant to a scalar parameter', () => {
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

  test('can add vectors', () => {
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
    const results = loaded_executable.execute([[client.bufferFromHostLiteral(argument1_literal), client.bufferFromHostLiteral(argument2_literal)]], {});

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual([4, 6]);
  });

  test('can add matrices', () => {
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
    const results = loaded_executable.execute([[client.bufferFromHostLiteral(argument1_literal), client.bufferFromHostLiteral(argument2_literal)]], {});

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual([8, 10, 12, 14, 16, 18]);
    expect(result_literal.shape().dimensions()).toStrictEqual([3, 2]);
  });

  test('can multiply matrices', () => {
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
    const results = loaded_executable.execute([[client.bufferFromHostLiteral(argument1_literal), client.bufferFromHostLiteral(argument2_literal)]], {});

    // Extract the resulting scalar.
    const result_buffer = results[0][0];
    const result_literal = result_buffer.toLiteralSync();

    expect(result_literal.shape().dimensions()).toStrictEqual([2, 2]);
    expect(result_literal.data(xla.PrimitiveType.F32)).toStrictEqual([58, 64, 139, 154]);
  });
});

describe('XLA literal', () => {
  test('can broadcast scalar to vector', () => {
    const shape = xla.Shape.forArray(xla.PrimitiveType.F32, [3]);
    const lit = xla.Literal.createR0(xla.PrimitiveType.F32, 1.0).broadcast(shape, []);
    expect(lit.data(xla.PrimitiveType.F32)).toStrictEqual([1, 1, 1]);
  });
});

