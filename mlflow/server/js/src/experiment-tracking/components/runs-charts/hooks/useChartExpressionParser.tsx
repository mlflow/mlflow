import { last, uniq } from 'lodash';
import type { RunsChartsLineChartExpression } from '../runs-charts.types';
import { useCallback } from 'react';

const VARIABLE_OPERATOR = '$';
const MATCH_VARIABLE_REGEX = /\$\{([^}]+)\}/g;

enum Operator {
  ADD = '+',
  SUBTRACT = '-',
  MULTIPLY = '*',
  DIVIDE = '/',
  POWER = '^',
  // unary operators
  NEGATIVE_SIGN = '_',
  POSITIVE_SIGN = '|',
}
enum Parenthesis {
  OPEN = '(',
  CLOSE = ')',
}
enum VariableParenthesis {
  OPEN = '{',
  CLOSE = '}',
}
// Higher precedence value means higher priority
const precedence = {
  [Operator.ADD]: 1,
  [Operator.SUBTRACT]: 1,
  [Operator.MULTIPLY]: 2,
  [Operator.DIVIDE]: 2,
  [Operator.POWER]: 3,
  [Operator.NEGATIVE_SIGN]: 4,
  [Operator.POSITIVE_SIGN]: 4,
};

const isBinaryOperator = (char: string | number | undefined) => {
  if (typeof char === 'string') {
    return [Operator.ADD, Operator.SUBTRACT, Operator.MULTIPLY, Operator.DIVIDE, Operator.POWER].includes(
      char as Operator,
    );
  }
  return false;
};

const isUnaryOperator = (char: string | number | undefined) => {
  if (typeof char === 'string') {
    return [Operator.NEGATIVE_SIGN, Operator.POSITIVE_SIGN].includes(char as Operator);
  }
  return false;
};
const isHigherPrecedence = (op1: Operator, op2: Operator) => precedence[op1] > precedence[op2];

const indexToVariableString = (name: number) => `${VARIABLE_OPERATOR}{${name}}`;
const variableStringToIndex = (variable: string) => variable.slice(2, -1);

// Parses variables from an expression and replace with an index
export const parseVariablesAndReplaceWithIndex = (expression: string) => {
  const matches = expression.match(MATCH_VARIABLE_REGEX);
  if (!matches) {
    return { expression, variables: [] };
  }
  // De-duplicate matches and map them to their index
  const dedupMatches = uniq(matches);
  const matchesMap: Record<string, string> = {};
  dedupMatches.forEach((match, index) => {
    matchesMap[match] = indexToVariableString(index);
  });
  // Convert each variable into an index number e.g. ${train/loss} => ${0}
  const replacedExpression = expression.replace(MATCH_VARIABLE_REGEX, (match) => {
    if (match in matchesMap) {
      return matchesMap[match];
    }
    return match;
  });

  const variables = dedupMatches.map(variableStringToIndex);
  return { expression: replacedExpression, variables };
};

// Pop function that throws an error if the stack is empty
const popAndValidate = (stack: (string | number)[], operandCount: number[]): string | number => {
  const lastElement = stack.pop();
  if (lastElement === undefined) {
    throw new Error('Invalid expression: stack is empty');
  } else if (isBinaryOperator(lastElement)) {
    if (operandCount[operandCount.length - 1] < 2) {
      throw new Error('Invalid expression: Stack has binary operator without enough operands');
    }
    operandCount[operandCount.length - 1]--;
  } else if (isUnaryOperator(lastElement)) {
    if (operandCount[operandCount.length - 1] < 1) {
      throw new Error('Invalid expression: Stack has unary operator without enough operands');
    }
  } else {
    throw new Error('Invalid expression: Stack has invalid elements');
  }
  return lastElement;
};

// Flushes all the unary operators on a given value
const flushUnaryOperators = (stack: (string | number)[], output: (string | number)[], operandCount: number[]) => {
  while (stack.length > 0 && isUnaryOperator(last(stack))) {
    output.push(popAndValidate(stack, operandCount));
  }
};

const toRPN = (expression: string) => {
  const stack: (string | number)[] = [];
  const output: (string | number)[] = [];
  const operandCount: number[] = [0];

  const incrementOperand = () => {
    operandCount[operandCount.length - 1]++;
  };
  for (let i = 0; i < expression.length; i++) {
    let char = expression[i];

    // Convert unary operators to placeholder negative and positive signs
    const isUnarySign = i === 0 || isBinaryOperator(expression[i - 1]) || expression[i - 1] === Parenthesis.OPEN;
    if (char === Operator.SUBTRACT && isUnarySign) {
      char = Operator.NEGATIVE_SIGN;
    } else if (char === Operator.ADD && isUnarySign) {
      char = Operator.POSITIVE_SIGN;
    }

    if (char === VARIABLE_OPERATOR) {
      let variable = '';
      if (i + 1 >= expression.length || expression[i + 1] !== VariableParenthesis.OPEN) {
        throw new Error('Invalid expression: Variable must be followed by {');
      }
      i++; // Skip '{'
      while (i + 1 < expression.length && expression[i + 1] !== VariableParenthesis.CLOSE) {
        variable += expression[++i];
      }
      i++; // Skip '}'
      output.push(variable);
      incrementOperand();
      flushUnaryOperators(stack, output, operandCount);
    } else if (/\d/.test(char) || char === '.') {
      // If the character is part of a number (digit or decimal point)
      let num = char;
      // Parse full number
      while (i + 1 < expression.length && (/\d/.test(expression[i + 1]) || expression[i + 1] === '.')) {
        num += expression[++i];
      }
      const periodMatches = num.match(/\./g);
      if (periodMatches && periodMatches.length > 1) {
        throw new Error('Invalid expression: Number has multiple decimal points');
      }
      const floatNum = parseFloat(num);
      incrementOperand();
      output.push(floatNum);
      flushUnaryOperators(stack, output, operandCount);
    } else if (isUnaryOperator(char)) {
      stack.push(char);
    } else if (isBinaryOperator(char)) {
      // If its a binary operator, we should have at least on element in the output to operate on
      if (output.length === 0) {
        throw new Error('Invalid expression: Binary operator without operands');
      }
      while (
        stack.length > 0 &&
        isBinaryOperator(last(stack)) &&
        (isHigherPrecedence(last(stack) as Operator, char as Operator) ||
          (last(stack) === char && char !== Operator.POWER))
      ) {
        output.push(popAndValidate(stack, operandCount));
      }
      stack.push(char);
    } else if (char === Parenthesis.OPEN) {
      stack.push(char);
      operandCount.push(0);
    } else if (char === Parenthesis.CLOSE) {
      while (stack.length > 0 && last(stack) !== Parenthesis.OPEN) {
        output.push(popAndValidate(stack, operandCount));
      }
      const openParen = stack.pop(); // Remove '(' from the stack
      if (openParen !== Parenthesis.OPEN) {
        throw new Error('Invalid expression: Parenthesis mismatch');
      }
      if (operandCount[operandCount.length - 1] !== 1) {
        throw new Error('Invalid expression: Parenthesis does not have exactly one operand');
      }
      operandCount.pop();
      incrementOperand();
      flushUnaryOperators(stack, output, operandCount);
    } else {
      throw new Error('Invalid expression: Unknown character in expression');
    }
  }
  // The stack should only have unary and binary operators at the end
  while (stack.length > 0) {
    output.push(popAndValidate(stack, operandCount));
  }
  if (operandCount.length !== 1 || operandCount[0] !== 1) {
    throw new Error('Invalid expression: Invalid number of operands');
  }
  return output;
};

const fromRPN = (tokens: (string | number)[]) => {
  const stack: (number | string)[] = [];
  tokens.forEach((token) => {
    if (typeof token === 'number') {
      stack.push(token);
      return;
    }
    if (isUnaryOperator(token)) {
      const x = stack.pop();
      if (typeof x !== 'number') {
        throw new Error('Invalid expression: Unary operator without operand');
      }
      switch (token) {
        case Operator.NEGATIVE_SIGN:
          stack.push(-x);
          break;
        case Operator.POSITIVE_SIGN:
          stack.push(x);
          break;
      }
    } else if (isBinaryOperator(token)) {
      const b = stack.pop();
      const a = stack.pop();
      if (typeof a !== 'number' || typeof b !== 'number') {
        throw new Error('Invalid expression: Binary operator without enough operands');
      }
      switch (token) {
        case Operator.ADD:
          stack.push(a + b);
          break;
        case Operator.SUBTRACT:
          stack.push(a - b);
          break;
        case Operator.MULTIPLY:
          stack.push(a * b);
          break;
        case Operator.DIVIDE:
          stack.push(a / b);
          break;
        case Operator.POWER:
          stack.push(Math.pow(a, b));
          break;
      }
    } else {
      throw new Error('Invalid expression: Unknown token in expression');
    }
  });
  if (stack.length !== 1 || typeof stack[0] !== 'number') {
    throw new Error('Invalid expression: Invalid expression result');
  }
  return stack[0];
};

export const useChartExpressionParser = () => {
  const compileExpression = useCallback(
    (expression: string, metricKeyList: string[]): RunsChartsLineChartExpression | undefined => {
      try {
        // Validate only contains valid characters
        const noVariableExpression = expression.replace(MATCH_VARIABLE_REGEX, '');
        if (!/^[0-9+\-*/().\s^]*$/.test(noVariableExpression)) {
          return undefined;
        }
        // Parse variables from expression and remove whitespace elsewhere
        const { expression: parsedExpression, variables } = parseVariablesAndReplaceWithIndex(expression);
        // Check if all variables are valid
        for (const variable of variables) {
          if (!metricKeyList.includes(variable)) {
            return undefined;
          }
        }
        const cleanedExpression = parsedExpression.replace(/\s/g, '');
        const replacedExpression = cleanedExpression.replace(MATCH_VARIABLE_REGEX, (match) => {
          const index = parseInt(match.slice(2, -1), 10);
          return `${VARIABLE_OPERATOR}{${variables[index]}}`;
        });
        // Convert expression to RPN
        const rpn = toRPN(replacedExpression);
        return {
          rpn,
          variables,
          expression,
        };
      } catch (e) {
        // If not a valid expression, return undefined
        return undefined;
      }
    },
    [],
  );

  const evaluateExpression = (
    chartExpression: RunsChartsLineChartExpression | undefined,
    variables: Record<string, number>,
  ): number | undefined => {
    if (chartExpression === undefined) {
      return undefined;
    }
    try {
      const parsedRPN = chartExpression.rpn.map((token) => {
        if (typeof token === 'string' && chartExpression.variables.includes(token)) {
          return variables[token];
        }
        return token;
      });
      return fromRPN(parsedRPN);
    } catch (e) {
      return undefined;
    }
  };

  return {
    compileExpression,
    evaluateExpression,
  };
};
