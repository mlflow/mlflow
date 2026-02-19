import { useCallback, useState } from 'react';
import type { MessageDescriptor } from 'react-intl';
import { defineMessage } from 'react-intl';

// Hardcoded model parameter definitions
const parameterDefinitions: {
  name: 'temperature' | 'max_tokens' | 'stop';
  type: 'slider' | 'input' | 'list';
  string: MessageDescriptor;
  helpString: MessageDescriptor;
  max?: number;
  min?: number;
  step?: number;
}[] = [
  {
    type: 'slider',
    name: 'temperature',
    string: defineMessage({
      defaultMessage: 'Temperature',
      description: 'Experiment page > prompt lab > temperature parameter label',
    }),
    helpString: defineMessage({
      defaultMessage: 'Increase or decrease the confidence level of the language model.',
      description: 'Experiment page > prompt lab > temperature parameter help text',
    }),
    max: 1,
    min: 0,
    step: 0.01,
  },
  {
    type: 'input',
    name: 'max_tokens',
    string: defineMessage({
      defaultMessage: 'Max tokens',
      description: 'Experiment page > prompt lab > max tokens parameter label',
    }),
    helpString: defineMessage({
      defaultMessage: 'Maximum number of language tokens returned from evaluation.',
      description: 'Experiment page > prompt lab > max tokens parameter help text',
    }),
    max: 64 * 1024,
    min: 1,
    step: 1,
  },
  {
    type: 'list',
    name: 'stop',
    string: defineMessage({
      defaultMessage: 'Stop Sequences',
      description: 'Experiment page > prompt lab > stop parameter label',
    }),
    helpString: defineMessage({
      defaultMessage: 'Specify sequences that signal the model to stop generating text.',
      description: 'Experiment page > prompt lab > stop parameter help text',
    }),
  },
];

// TODO: Fetch better values for default parameters
const DEFAULT_PARAMETER_VALUES = {
  temperature: 0.01,
  max_tokens: 100,
};

export const usePromptEvaluationParameters = () => {
  const [parameters, updateParameters] = useState<{
    temperature: number;
    max_tokens: number;
    stop?: string[] | undefined;
  }>(DEFAULT_PARAMETER_VALUES);

  const updateParameter = useCallback((name: string, value: number | string[]) => {
    updateParameters((currentParameters) => ({ ...currentParameters, [name]: value }));
  }, []);

  return {
    parameterDefinitions,
    parameters,
    updateParameter,
  };
};
