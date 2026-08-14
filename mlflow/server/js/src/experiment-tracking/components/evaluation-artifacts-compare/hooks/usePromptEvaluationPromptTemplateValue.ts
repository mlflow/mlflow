import React, { useCallback, useRef, useState } from 'react';

import type { TextAreaRef } from '@databricks/design-system';
import {
  DEFAULT_PROMPTLAB_NEW_TEMPLATE_VALUE,
  extractPromptInputVariables,
} from '../../prompt-engineering/PromptEngineering.utils';
import { max } from 'lodash';

const newVariableStartSegment = ' {{ ';
const newVariableEndSegment = ' }}';
const newDefaultVariableName = 'new_variable';

const getNewVariableName = (alreadyExistingVariableNames: string[] = []) => {
  if (!alreadyExistingVariableNames.includes(newDefaultVariableName)) {
    return newDefaultVariableName;
  }

  const maximumVariableNameIndex =
    max(alreadyExistingVariableNames.map((name) => parseInt(name.match(/new_variable_(\d+)/)?.[1] || '1', 10))) || 1;

  return `${newDefaultVariableName}_${maximumVariableNameIndex + 1}`;
};

/**
 * Keeps track of the current prompt value and exports method for adding + autoselecting new variables
 */
export const usePromptEvaluationPromptTemplateValue = () => {
  const [promptTemplate, updatePromptTemplate] = useState(DEFAULT_PROMPTLAB_NEW_TEMPLATE_VALUE);

  const promptTemplateRef = useRef<HTMLTextAreaElement>();

  const handleAddVariableToTemplate = useCallback(() => {
    updatePromptTemplate((template) => {
      const newVariableName = getNewVariableName(extractPromptInputVariables(template));
      const newValue = `${template}${newVariableStartSegment}${newVariableName}${newVariableEndSegment}`;

      // Wait until the next execution frame
      requestAnimationFrame(() => {
        const textAreaElement = promptTemplateRef.current;
        if (!textAreaElement) {
          return;
        }
        // Focus the element and set the newly added variable name
        textAreaElement.focus();
        textAreaElement.setSelectionRange(
          newValue.length - newVariableName.length - newVariableEndSegment.length,
          newValue.length - newVariableEndSegment.length,
        );
      });
      return newValue;
    });
  }, [updatePromptTemplate]);

  const savePromptTemplateInputRef = useCallback((ref: TextAreaRef) => {
    promptTemplateRef.current = ref?.resizableTextArea?.textArea;
  }, []);

  return {
    savePromptTemplateInputRef,
    handleAddVariableToTemplate,
    promptTemplate,
    updatePromptTemplate,
  };
};
