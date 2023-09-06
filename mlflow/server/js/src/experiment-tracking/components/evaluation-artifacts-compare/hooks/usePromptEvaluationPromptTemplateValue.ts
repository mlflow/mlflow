import React, { useCallback, useRef, useState } from 'react';

import type { TextAreaRef } from 'antd/lib/input/TextArea';
import { DEFAULT_PROMPTLAB_NEW_TEMPLATE_VALUE } from '../../prompt-engineering/PromptEngineering.utils';

const newVariableStartSegment = ' {{ ';
const newVariableEndSegment = ' }}';
const newVariableNameSegment = 'new_variable';

/**
 * Keeps track of the current prompt value and exports method for adding + autoselecting new variables
 */
export const usePromptEvaluationPromptTemplateValue = () => {
  const [promptTemplate, updatePromptTemplate] = useState(DEFAULT_PROMPTLAB_NEW_TEMPLATE_VALUE);

  const promptTemplateRef = useRef<HTMLTextAreaElement>();

  const handleAddVariableToTemplate = useCallback(() => {
    updatePromptTemplate((template) => {
      const newValue = `${template}${newVariableStartSegment}${newVariableNameSegment}${newVariableEndSegment}`;

      // Wait until the next execution frame
      requestAnimationFrame(() => {
        const textAreaElement = promptTemplateRef.current;
        if (!textAreaElement) {
          return;
        }
        // Focus the element and set the newly added variable name
        textAreaElement.focus();
        textAreaElement.setSelectionRange(
          newValue.length - newVariableNameSegment.length - newVariableEndSegment.length,
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
