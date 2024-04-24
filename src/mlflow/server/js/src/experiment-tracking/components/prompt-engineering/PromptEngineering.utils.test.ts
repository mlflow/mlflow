import {
  canEvaluateOnRun,
  compilePromptInputText,
  extractEvaluationPrerequisitesForRun,
  extractPromptInputVariables,
  getPromptInputVariableNameViolations,
} from './PromptEngineering.utils';

describe('PromptEngineering utils', () => {
  describe('extractPromptInputVariables', () => {
    test('correctly extracts multiple variable names', () => {
      const variables = extractPromptInputVariables('this {{a}} is a {{b}} test but {{a}} is duplicated');
      expect(variables).toEqual(['a', 'b']);
    });
    test('correctly handles spaces in the variable name', () => {
      const variables = extractPromptInputVariables('this {{a  }} is a {{   b}} test but {{ cc }}');
      expect(variables).toEqual(['a', 'b', 'cc']);
    });
    test('correctly handles malformed templates with not matching brackets', () => {
      expect(extractPromptInputVariables('this {{a} asdafsdf')).toEqual([]);
      expect(extractPromptInputVariables('this {a}} asdafsdf')).toEqual([]);
    });
    test('correctly handles malformed templates with spaces inside', () => {
      expect(extractPromptInputVariables('this {{a ww}} asdafsdf')).toEqual([]);
    });
    test('correctly handles letter case', () => {
      expect(extractPromptInputVariables('this {{ONE}} parameter and another {{One}} parameter')).toEqual(['one']);
    });
  });

  describe('extractEvaluationPrerequisitesForRun', () => {
    test('correctly extracts existing values', () => {
      const evaluationPrerequisites = extractEvaluationPrerequisitesForRun({
        runUuid: 'test-run',
        params: [
          { key: 'model_route', value: 'test-model-route' },
          { key: 'prompt_template', value: 'test-prompt-template' },
          { key: 'max_tokens', value: '1000' },
          { key: 'temperature', value: '0.5' },
          { key: 'stop', value: '[END]' },
        ],
      } as any);
      expect(evaluationPrerequisites).toEqual({
        parameters: {
          max_tokens: 1000,
          temperature: 0.5,
          stop: ['END'],
        },
        promptTemplate: 'test-prompt-template',
        routeName: 'test-model-route',
      });
    });
    test('correctly extracts existing with empty stop sequence', () => {
      const evaluationPrerequisites = extractEvaluationPrerequisitesForRun({
        runUuid: 'test-run',
        params: [{ key: 'stop', value: '[]' }],
      } as any);
      expect(evaluationPrerequisites).toEqual({
        parameters: {
          stop: [],
        },
      });
    });
    test('correctly handles missing and invalid values', () => {
      const evaluationPrerequisites = extractEvaluationPrerequisitesForRun({
        runUuid: 'test-run',
        params: [
          { key: 'max_tokens', value: 'some-invalid-value' },
          { key: 'temperature', value: 'some-invalid-value' },
        ],
      } as any);
      expect(evaluationPrerequisites).toEqual({
        parameters: {
          max_tokens: undefined,
          temperature: undefined,
          stop: undefined,
        },
        promptTemplate: undefined,
        routeName: undefined,
      });
    });
  });
  describe('canEvaluateOnRun', () => {
    const promptRun: any = {
      tags: {
        'mlflow.runSourceType': { key: 'mlflow.runSourceType', value: 'PROMPT_ENGINEERING' },
      },
    };
    const genericRun: any = {
      tags: {
        'mlflow.runSourceType': { key: 'mlflow.runSourceType', value: 'some-other-run-source' },
      },
    };
    it('correctly determines evaluateable runs', () => {
      expect(canEvaluateOnRun(promptRun)).toBeTruthy();
      expect(canEvaluateOnRun(genericRun)).toBeFalsy();
    });
  });
  describe('compilePromptInputText', () => {
    test('correctly compiles the input text', () => {
      expect(
        compilePromptInputText('my name is {{ name }} {{ surname }}', {
          name: 'John',
          surname: 'Johnson',
        }),
      ).toEqual('my name is John Johnson');
    });

    test('correctly handles the casing', () => {
      expect(
        compilePromptInputText('my name is {{ nAmE }}', {
          NaMe: 'John',
        }),
      ).toEqual('my name is John');
    });

    test('correctly handles loosely formatted brackets', () => {
      expect(
        compilePromptInputText('my name is {{   name     }} and surname is {{    surname}}', {
          name: 'John',
          surname: 'Johnson',
        }),
      ).toEqual('my name is John and surname is Johnson');
    });

    test('correctly handles fields with no values', () => {
      expect(
        compilePromptInputText('the value is {{ value }} and not provided value is {{ not_provided }}', {
          value: 'TestValue',
        }),
      ).toEqual('the value is TestValue and not provided value is {{ not_provided }}');
    });
  });
  describe('getPromptInputVariableNameViolations', () => {
    it('should return nothing on valid template', () => {
      expect(getPromptInputVariableNameViolations('this is {{var_a}} and {{var_b}}')).toEqual({
        namesWithSpaces: [],
      });
    });
    it('should report invalid names with spaces inside', () => {
      expect(
        getPromptInputVariableNameViolations(
          'this is {{invalid var a}} and {{valid_var_b}} and {{    invalid-var c  }} and {{ valid-var-d }} and {{    invalid-var named_e  }}',
        ),
      ).toEqual({
        namesWithSpaces: ['invalid var a', 'invalid-var c', 'invalid-var named_e'],
      });
    });
  });
});
