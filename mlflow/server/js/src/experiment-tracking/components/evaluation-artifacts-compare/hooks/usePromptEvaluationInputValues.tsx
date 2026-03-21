import { useCallback, useEffect, useMemo, useState } from 'react';
import { debounce, fromPairs, isEqual } from 'lodash';
import {
  DEFAULT_PROMPTLAB_INPUT_VALUES,
  extractPromptInputVariables,
  getPromptInputVariableNameViolations,
} from '../../prompt-engineering/PromptEngineering.utils';

export const usePromptEvaluationInputValues = () => {
  const [inputVariables, updateInputVariablesDirect] = useState<string[]>(extractPromptInputVariables(''));

  const [inputVariableNameViolations, setInputVariableNameViolations] = useState<
    ReturnType<typeof getPromptInputVariableNameViolations>
  >({ namesWithSpaces: [] });

  const [inputVariableValues, updateInputVariableValues] =
    useState<Record<string, string>>(DEFAULT_PROMPTLAB_INPUT_VALUES);

  const clearInputVariableValues = useCallback(() => updateInputVariableValues({}), []);

  const updateInputVariables = useMemo(
    () =>
      // Prevent calculating new input variable set on every keystroke of a template,
      // let's debounce it by 250ms
      debounce((promptTemplate: string) => {
        updateInputVariablesDirect((currentInputVariables) => {
          const newInputVariables = extractPromptInputVariables(promptTemplate);
          if (!isEqual(newInputVariables, currentInputVariables)) {
            return newInputVariables;
          }
          return currentInputVariables;
        });
        setInputVariableNameViolations(getPromptInputVariableNameViolations(promptTemplate));
      }, 250),
    [],
  );

  const updateInputVariableValue = useCallback((name: string, value: string) => {
    updateInputVariableValues((values) => ({ ...values, [name]: value }));
  }, []);

  // Sanitize the variable dictionary so only actually used variables
  // will be returned (discard leftovers from previous prompt templates)
  const sanitizedInputVariableValues = useMemo(
    () => fromPairs(Object.entries(inputVariableValues).filter(([key]) => inputVariables.includes(key))),
    [inputVariableValues, inputVariables],
  );

  return {
    updateInputVariables,
    inputVariables,
    inputVariableValues: sanitizedInputVariableValues,
    updateInputVariableValue,
    inputVariableNameViolations,
    clearInputVariableValues,
  };
};
