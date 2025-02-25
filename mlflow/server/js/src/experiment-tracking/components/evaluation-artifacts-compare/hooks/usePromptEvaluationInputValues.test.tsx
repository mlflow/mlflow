import { renderHook, act } from '@testing-library/react';
import { usePromptEvaluationInputValues } from './usePromptEvaluationInputValues';

jest.useFakeTimers();

describe('usePromptEvaluationInputValues', () => {
  const mountTestComponent = () => {
    const { result } = renderHook(() => usePromptEvaluationInputValues());
    return { getHookResult: () => result.current };
  };

  it('should properly extract and update variables', () => {
    const { getHookResult } = mountTestComponent();

    const { updateInputVariableValue, updateInputVariables } = getHookResult();

    // Assert the initial variable set
    expect(getHookResult().inputVariables).toEqual([]);

    // Update the prompt template
    updateInputVariables('a variable {{a}} and the other one {{b}}');

    // Checking immediately, we should still be seeing the old one
    expect(getHookResult().inputVariables).toEqual([]);

    // Wait for .5s
    act(() => {
      jest.advanceTimersByTime(500);
    });

    // We should have the input variable list updated
    expect(getHookResult().inputVariables).toEqual(expect.arrayContaining(['a', 'b']));

    // Set and assert some values
    act(() => {
      updateInputVariableValue('a', 'value of a');
      updateInputVariableValue('b', 'value of b');
    });

    expect(getHookResult().inputVariableValues).toEqual({ a: 'value of a', b: 'value of b' });
  });

  it('should properly report variable name violations', async () => {
    const { getHookResult } = mountTestComponent();

    const { updateInputVariables } = getHookResult();

    updateInputVariables('a variable {{a}} and the other one {{b}}');

    // Wait for .5s
    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    expect(getHookResult().inputVariableNameViolations).toEqual({ namesWithSpaces: [] });

    await act(async () => {
      updateInputVariables('a variable {{a}} and the other one {{x y z}}');
    });

    // Wait for .5s
    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    expect(getHookResult().inputVariableNameViolations).toEqual({ namesWithSpaces: ['x y z'] });
  });
});
