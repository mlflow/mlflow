import { act, renderHook } from '@testing-library/react';
import { usePromptEvaluationPromptTemplateValue } from './usePromptEvaluationPromptTemplateValue';
import { extractPromptInputVariables } from '../../prompt-engineering/PromptEngineering.utils';

describe('usePromptEvaluationPromptTemplateValue', () => {
  test('should properly generate subsequent template variable names', async () => {
    const { result } = renderHook(() => usePromptEvaluationPromptTemplateValue());
    expect(result.current.promptTemplate).toBeTruthy();

    await act(async () => {
      result.current.updatePromptTemplate('some new prompt template {{my_var}}');
    });

    expect(result.current.promptTemplate).toEqual('some new prompt template {{my_var}}');

    await act(async () => {
      result.current.handleAddVariableToTemplate();
      result.current.handleAddVariableToTemplate();
      result.current.handleAddVariableToTemplate();
    });

    expect(extractPromptInputVariables(result.current.promptTemplate)).toEqual([
      'my_var',
      'new_variable',
      'new_variable_2',
      'new_variable_3',
    ]);
  });
});
