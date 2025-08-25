/* eslint-disable no-template-curly-in-string */
// Disabled because of the following lint error: Unexpected template string expression

import { renderHook } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { useChartExpressionParser, parseVariablesAndReplaceWithIndex } from './useChartExpressionParser';

describe('parseVariablesAndReplaceWithIndex', () => {
  it('should handle variables in expression', () => {
    expect(parseVariablesAndReplaceWithIndex('${abcde12345} + 4')).toEqual({
      variables: ['abcde12345'],
      expression: '${0} + 4',
    });
    // Handles duplicates
    expect(parseVariablesAndReplaceWithIndex('${abcde12345} + 4 - -((${abcde12345}))')).toEqual({
      variables: ['abcde12345'],
      expression: '${0} + 4 - -((${0}))',
    });
    // Handles spacing
    expect(parseVariablesAndReplaceWithIndex('2 * -${a test 1234/this can be pretty log\t\n}')).toEqual({
      variables: ['a test 1234/this can be pretty log\t\n'],
      expression: '2 * -${0}',
    });

    // Many variables
    expect(
      parseVariablesAndReplaceWithIndex('${a} + ${b} * ${c} - ${d} / ${e} + ${f} + ${g} + ${h} + ${i} + ${k} + ${a}'),
    ).toEqual({
      variables: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k'],
      expression: '${0} + ${1} * ${2} - ${3} / ${4} + ${5} + ${6} + ${7} + ${8} + ${9} + ${0}',
    });

    // All special characters
    expect(
      parseVariablesAndReplaceWithIndex(
        '${This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).}',
      ),
    ).toEqual({
      variables: [
        'This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).',
      ],
      expression: '${0}',
    });

    expect(parseVariablesAndReplaceWithIndex('${}')).toEqual({ expression: '${}', variables: [] });
  });
});

describe('useChartExpressionParser', () => {
  const { result } = renderHook(() => useChartExpressionParser());
  const { compileExpression, evaluateExpression } = result.current;

  it('should handle variables in compilation', () => {
    expect(compileExpression('${x}+2', ['x'])).toEqual({ rpn: ['x', 2, '+'], variables: ['x'], expression: '${x}+2' });
    expect(compileExpression('1+${x}', ['x'])).toEqual({ rpn: [1, 'x', '+'], variables: ['x'], expression: '1+${x}' });
    expect(compileExpression('${x}-2', ['x'])).toEqual({ rpn: ['x', 2, '-'], variables: ['x'], expression: '${x}-2' });
  });

  it('should handle multiple variables in compilation', () => {
    expect(compileExpression('${x}+${y}', ['x', 'y'])).toEqual({
      rpn: ['x', 'y', '+'],
      variables: ['x', 'y'],
      expression: '${x}+${y}',
    });
    expect(compileExpression('${x}-${y}', ['x', 'y'])).toEqual({
      rpn: ['x', 'y', '-'],
      variables: ['x', 'y'],
      expression: '${x}-${y}',
    });
    expect(compileExpression('${x}*${y}', ['x', 'y'])).toEqual({
      rpn: ['x', 'y', '*'],
      variables: ['x', 'y'],
      expression: '${x}*${y}',
    });
    expect(compileExpression('${x}/${y}', ['x', 'y'])).toEqual({
      rpn: ['x', 'y', '/'],
      variables: ['x', 'y'],
      expression: '${x}/${y}',
    });
    expect(compileExpression('${x}^${y}', ['x', 'y'])).toEqual({
      rpn: ['x', 'y', '^'],
      variables: ['x', 'y'],
      expression: '${x}^${y}',
    });
  });

  it('should handle multiple variables in complex functions', () => {
    expect(compileExpression('${x}+${y}*${z}', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', 'z', '*', '+'],
      variables: ['x', 'y', 'z'],
      expression: '${x}+${y}*${z}',
    });
    expect(compileExpression('${x}*${y}+${z}', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', '*', 'z', '+'],
      variables: ['x', 'y', 'z'],
      expression: '${x}*${y}+${z}',
    });
    expect(compileExpression('${x}+${y}^${z}', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', 'z', '^', '+'],
      variables: ['x', 'y', 'z'],
      expression: '${x}+${y}^${z}',
    });
    expect(compileExpression('${x}^${y}+${z}', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', '^', 'z', '+'],
      variables: ['x', 'y', 'z'],
      expression: '${x}^${y}+${z}',
    });
    expect(compileExpression('${x}*${y}^${z}', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', 'z', '^', '*'],
      variables: ['x', 'y', 'z'],
      expression: '${x}*${y}^${z}',
    });
    expect(compileExpression('${x}^${y}*${z}', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', '^', 'z', '*'],
      variables: ['x', 'y', 'z'],
      expression: '${x}^${y}*${z}',
    });

    // With Parenthesis
    expect(compileExpression('(${x}+${y})*${z}', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', '+', 'z', '*'],
      variables: ['x', 'y', 'z'],
      expression: '(${x}+${y})*${z}',
    });
    expect(compileExpression('${x}+(${y}*${z})', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', 'z', '*', '+'],
      variables: ['x', 'y', 'z'],
      expression: '${x}+(${y}*${z})',
    });
    expect(compileExpression('(${x}+${y})*${z}', ['x', 'y', 'z'])).toEqual({
      rpn: ['x', 'y', '+', 'z', '*'],
      variables: ['x', 'y', 'z'],
      expression: '(${x}+${y})*${z}',
    });

    // With negative variable values
    expect(compileExpression('${x}+-${y}', ['x', 'y'])).toEqual({
      rpn: ['x', 'y', '_', '+'],
      variables: ['x', 'y'],
      expression: '${x}+-${y}',
    });
    expect(compileExpression('${x}--${y}', ['x', 'y'])).toEqual({
      rpn: ['x', 'y', '_', '-'],
      variables: ['x', 'y'],
      expression: '${x}--${y}',
    });

    // With floating point values and negative variables
    expect(
      compileExpression('${abcde#12349!o++++----////*^} + ${text} * 0.2', ['abcde#12349!o++++----////*^', 'text']),
    ).toEqual({
      rpn: ['abcde#12349!o++++----////*^', 'text', 0.2, '*', '+'],
      variables: ['abcde#12349!o++++----////*^', 'text'],
      expression: '${abcde#12349!o++++----////*^} + ${text} * 0.2',
    });
  });

  it('should handle invalid expressions', () => {
    // Invalid variables
    expect(compileExpression('${x}+${y}', ['x'])).toEqual(undefined);
    expect(compileExpression('${x}+${y}', ['y'])).toEqual(undefined);

    // Invalid evaluation
    expect(compileExpression('*', [])).toEqual(undefined);
    expect(compileExpression('-', [])).toEqual(undefined);
    expect(compileExpression('/', [])).toEqual(undefined);
    expect(compileExpression('^', [])).toEqual(undefined);
    expect(compileExpression('+', [])).toEqual(undefined);
    expect(compileExpression('(', [])).toEqual(undefined);
    expect(compileExpression(')', [])).toEqual(undefined);
    expect(compileExpression('${', [])).toEqual(undefined);
    expect(compileExpression('${}', [])).toEqual(undefined);
    expect(compileExpression('${}+2', [])).toEqual(undefined);
    expect(compileExpression('*1', [])).toEqual(undefined);
    expect(compileExpression('1*', [])).toEqual(undefined);
    expect(compileExpression('1+', [])).toEqual(undefined);
    expect(compileExpression('1+', [])).toEqual(undefined);
    expect(compileExpression('1-', [])).toEqual(undefined);
    expect(compileExpression('1/', [])).toEqual(undefined);
    expect(compileExpression('/1', [])).toEqual(undefined);
    expect(compileExpression('1^', [])).toEqual(undefined);
    expect(compileExpression('^1', [])).toEqual(undefined);
    expect(compileExpression('1(', [])).toEqual(undefined);
    expect(compileExpression(')1', [])).toEqual(undefined);
    expect(compileExpression('1)', [])).toEqual(undefined);

    // Invalid evaluation with parenthesis
    expect(compileExpression('(', [])).toEqual(undefined);
    expect(compileExpression(')', [])).toEqual(undefined);
    expect(compileExpression('()', [])).toEqual(undefined);
    expect(compileExpression('1+(', [])).toEqual(undefined);
    expect(compileExpression('1+)', [])).toEqual(undefined);
    expect(compileExpression('1+(1', [])).toEqual(undefined);
    expect(compileExpression('1+1)', [])).toEqual(undefined);
    expect(compileExpression('123 - (', [])).toEqual(undefined);
    expect(compileExpression('123 - ()', [])).toEqual(undefined);
    expect(compileExpression('123 - (-)', [])).toEqual(undefined);
  });

  const parseExpression = (expr: string) => {
    return evaluateExpression(compileExpression(expr, []), {});
  };

  it('should parse simple expressions', () => {
    expect(parseExpression('1+2')).toBe(3);
    expect(parseExpression('1-2')).toBe(-1);
    expect(parseExpression('1*2')).toBe(2);
    expect(parseExpression('1/2')).toBe(0.5);
    expect(parseExpression('1^2')).toBe(1);
  });

  it('should parse complex expressions', () => {
    expect(parseExpression('1+2*3')).toBe(7);
    expect(parseExpression('1*2+3')).toBe(5);
    expect(parseExpression('1+2^3')).toBe(9);
    expect(parseExpression('1^2+3')).toBe(4);
    expect(parseExpression('1*2^3')).toBe(8);
    expect(parseExpression('1^2*3')).toBe(3);
  });

  it('should handle parenthesis', () => {
    expect(parseExpression('(1+2)*3')).toBe(9);
    expect(parseExpression('1*(2+3)')).toBe(5);
    expect(parseExpression('(1+2)^3')).toBe(27);
    expect(parseExpression('1^(2+3)')).toBe(1);
    expect(parseExpression('1*(2^3)')).toBe(8);
    expect(parseExpression('1^(2*3)')).toBe(1);
  });

  it('should handle nested parenthesis', () => {
    expect(parseExpression('(1+(2*3))')).toBe(7);
    expect(parseExpression('((1+2)*3)')).toBe(9);
    expect(parseExpression('(1+(2^3))')).toBe(9);
    expect(parseExpression('(1^(2+3))')).toBe(1);
    expect(parseExpression('(1*(2^3))')).toBe(8);
    expect(parseExpression('(1^(2*3))')).toBe(1);
  });

  it('should handle complex nested parenthesis', () => {
    expect(parseExpression('((1+2)*3+(4/2))^2')).toBe(121);
    expect(parseExpression('((1+2)^3+(4/2))^2')).toBe(841);
    expect(parseExpression('((1+2)^(3+(4/2)))^2')).toBe(59049);
  });

  it('should handle whitespace', () => {
    expect(parseExpression(' 1 + 2 ')).toBe(3);
    expect(parseExpression('1 + 2 ')).toBe(3);
    expect(parseExpression(' 1 + 2')).toBe(3);
  });

  it('should handle invalid expressions (2)', () => {
    expect(parseExpression('1+')).toBe(undefined);
    expect(parseExpression('1+2+')).toBe(undefined);
    expect(parseExpression('1+2+3+')).toBe(undefined);
    expect(parseExpression('1+2+3+4+')).toBe(undefined);
    expect(parseExpression('1.1.1 + 2')).toBe(undefined);
    expect(parseExpression('1 + 2.2.2')).toBe(undefined);
    expect(parseExpression('1***2')).toBe(undefined);
    // Tests for unsupported operations or syntax
    expect(parseExpression('hello + 2')).toBe(undefined);
    expect(parseExpression('3 + world')).toBe(undefined);
    expect(parseExpression('1 + (2')).toBe(undefined);
    expect(parseExpression('1 + 2)')).toBe(undefined);
    expect(parseExpression('((1+2)')).toBe(undefined);
    expect(parseExpression('(1+2))')).toBe(undefined);

    // Tests for empty input and spaces only
    expect(parseExpression('')).toBe(undefined);
    expect(parseExpression('   ')).toBe(undefined);

    // Test for numbers combined with alphabets without operators
    expect(parseExpression('1a2')).toBe(undefined);
    expect(parseExpression('2b + 3')).toBe(undefined);

    // Test for invalid use of function names or similar characters
    expect(parseExpression('sin(30)')).toBe(undefined);
    expect(parseExpression('Math.max(10, 20)')).toBe(undefined);

    // Tests for incorrect use of parentheses and operators
    expect(parseExpression('1 + (2 * 3')).toBe(undefined);
    expect(parseExpression('(1 + 2 * 3))')).toBe(undefined);
    expect(parseExpression(')1 + 2(')).toBe(undefined);
    expect(parseExpression('1 + (2 + 3)) * 4')).toBe(undefined);

    // Tests for invalid operator combinations
    expect(parseExpression('1 + * 2')).toBe(undefined);
    expect(parseExpression('/ 2 + 3')).toBe(undefined);
    expect(parseExpression('1 **/ 2')).toBe(undefined);
    expect(parseExpression('1 $$ 2')).toBe(undefined);

    // Tests involving alphanumeric combinations
    expect(parseExpression('abc123 + 3')).toBe(undefined);
    expect(parseExpression('123xyz * 3')).toBe(undefined);
    expect(parseExpression('1_2 + 3')).toBe(undefined);

    // Tests with unexpected special characters
    expect(parseExpression('1 + 2 # 3')).toBe(undefined);
    expect(parseExpression('1 % 2')).toBe(undefined);
    expect(parseExpression('1 + 2@3')).toBe(undefined);
    expect(parseExpression('1 + !2')).toBe(undefined);

    // Test with leading and multiple operators
    expect(parseExpression('1 ** -+ 2')).toBe(undefined);

    // Tests for complex malformed expressions
    expect(parseExpression('(3 + 5)(2 + 2)')).toBe(undefined);
    expect(parseExpression('1 + {2*3}')).toBe(undefined);
  });

  it('should handle invalid parenthesis expressions', () => {
    expect(parseExpression('1+(2*3')).toBe(undefined);
    expect(parseExpression('(1+(2*3')).toBe(undefined);
    expect(parseExpression('((1+(2*3)')).toBe(undefined);
    expect(parseExpression('((1+(2*3))')).toBe(undefined);
    expect(parseExpression('()')).toBe(undefined);
    expect(parseExpression('(1+*)')).toBe(undefined);
  });

  it('should handle infinite and negative infinity', () => {
    expect(parseExpression('1/0')).toBe(Infinity);
    expect(parseExpression('1/((4 + 5)^0 - 1)')).toBe(Infinity);
    expect(parseExpression('-1/((4 + 5)^0 - 1)')).toBe(-Infinity);
    expect(parseExpression('1/-0')).toBe(-Infinity);
  });

  it('should handle negative numbers', () => {
    expect(parseExpression('-1+2')).toBe(1);
    expect(parseExpression('1+-2')).toBe(-1);
    expect(parseExpression('1+(-2)')).toBe(-1);
    expect(parseExpression('1-(-2)')).toBe(3);
    expect(parseExpression('1 + -2')).toBe(-1);
    expect(parseExpression('1--2')).toBe(3);
    expect(parseExpression('1---2')).toBe(-1);
    expect(parseExpression('1+--2')).toBe(3);
    expect(parseExpression('1+-(-2+-1)')).toBe(4);
    expect(parseExpression('1 + 2 * -3 / (4 - -5)')).toBeCloseTo(0.3333333);
    expect(parseExpression('-1 * (2 + 3) / 4')).toBe(-1.25);
    expect(parseExpression('(-1 + 2) * -3 / 4')).toBe(-0.75);
    expect(parseExpression('1 + -2 * 3 / (4 - -5)')).toBeCloseTo(0.3333333);
    expect(parseExpression('(-1 + 2) * (3 - 4 / 5)')).toBe(2.2);
    expect(parseExpression('1 + (2 + 3 * -4) / 5')).toBe(-1);
    expect(parseExpression('(-1 + 2 * 3) / (4 - -5)')).toBeCloseTo(0.555555);
    expect(parseExpression('-(2) + 3')).toBe(1);
    expect(parseExpression('5 * -3 + 2')).toBe(-13);
  });

  it('should handle positive numbers', () => {
    expect(parseExpression('1 ++2')).toBe(3);
  });

  it('should handle floating point numbers', () => {
    expect(parseExpression('1.1+2.2')).toBeCloseTo(3.3);
    expect(parseExpression('1.1-2.2')).toBeCloseTo(-1.1);
    expect(parseExpression('1.1*2.2')).toBeCloseTo(2.42);
    expect(parseExpression('1.1/2.2')).toBeCloseTo(0.5);
    expect(parseExpression('1.1^2.2')).toBeCloseTo(1.233);
  });

  it('should handle negative floating point numbers', () => {
    expect(parseExpression('-1.1+2.2')).toBeCloseTo(1.1);
    expect(parseExpression('1.1+-2.2')).toBeCloseTo(-1.1);
    expect(parseExpression('1.1+(-2.2)')).toBeCloseTo(-1.1);
    expect(parseExpression('1.1-(-2.2)')).toBeCloseTo(3.3);
  });
});
