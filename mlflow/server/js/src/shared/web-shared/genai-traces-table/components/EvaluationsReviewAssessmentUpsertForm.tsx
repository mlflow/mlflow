import { useMemo, useState } from 'react';

import {
  Button,
  Input,
  Spacer,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  useComboboxState,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { getEvaluationResultAssessmentValue } from './GenAiEvaluationTracesReview.utils';
import type {
  AssessmentDropdownSuggestionItem,
  RunEvaluationResultAssessment,
  RunEvaluationResultAssessmentDraft,
} from '../types';

/**
 * A form capable of adding or editing an assessment.
 */
export const EvaluationsReviewAssessmentUpsertForm = ({
  editedAssessment,
  valueSuggestions,
  onSave,
  onCancel,
  readOnly = false,
}: {
  editedAssessment?: RunEvaluationResultAssessment | RunEvaluationResultAssessmentDraft;
  valueSuggestions: AssessmentDropdownSuggestionItem[];
  onSave: (values: { value: string | boolean; rationale?: string; assessmentName?: string }) => void;
  onCancel: () => void;
  readOnly?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [rationale, setRationale] = useState<string | undefined>(() => {
    return editedAssessment?.rationale || undefined;
  });

  const [inputValue, setInputValue] = useState('');

  const [formValue, setFormValue] = useState<AssessmentDropdownSuggestionItem | undefined>(() => {
    // If we're editing an existing assessment, find relevant suggestion and use it as a form value
    if (editedAssessment) {
      const value = getEvaluationResultAssessmentValue(editedAssessment)?.toString();
      if (value) {
        return valueSuggestions.find((item) => item.key === value) ?? { key: value, label: value };
      }

      // Special case: if there's no value at all but we use a draft assessment, use assessment name
      return { key: editedAssessment.name, label: editedAssessment.name };
    }

    return { key: '', label: '' };
  });

  const [showAllSuggestions, setShowAllSuggestions] = useState(false);

  const filteredSuggestions = useMemo(
    () => valueSuggestions.filter((item) => item.label.toLowerCase().includes(inputValue.toLowerCase())),
    [inputValue, valueSuggestions],
  );

  // Show either all or filtered suggestions
  const visibleSuggestions = showAllSuggestions ? valueSuggestions : filteredSuggestions;

  // The combobox is displayed if there are suggestions or a custom value is provided
  const displayCombobox = inputValue || visibleSuggestions.length > 0;

  const comboboxState = useComboboxState<AssessmentDropdownSuggestionItem>({
    componentId:
      'codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewassessmentupsertform.tsx_124',
    allItems: valueSuggestions,
    items: visibleSuggestions,
    setItems: () => {},
    setInputValue: (val) => {
      setShowAllSuggestions(false);
      setInputValue(val);
    },
    multiSelect: false,
    allowNewValue: true,
    itemToString: (item) => (item ? item.label : ''),
    formValue,
    initialSelectedItem: formValue,
    initialInputValue: formValue?.label ?? '',
    formOnChange: (item) => {
      // If no changes made to the currently selected item, do nothing.
      // This is required, otherwise TypeaheadCombobox will replace object with plain string
      if (formValue?.label === item) {
        return;
      }

      // If provided custom value, construct a new item
      if (typeof item === 'string') {
        setFormValue({ key: inputValue ?? '', label: inputValue ?? '' });
        return;
      }

      // If used a dropdown option, set it as a form value
      setFormValue(item);
    },
    onIsOpenChange(isOpen) {
      if (isOpen) {
        // After uses clicks on the combobox, we're displaying all suggestions
        setShowAllSuggestions(true);
      }
    },
  });

  const addNewElementLabel = intl.formatMessage(
    {
      defaultMessage: 'Add "{label}"',
      description: 'Evaluation review > assessments > add new custom value element',
    },
    {
      label: inputValue,
    },
  );

  return (
    <div>
      <TypeaheadComboboxRoot comboboxState={comboboxState}>
        <TypeaheadComboboxInput
          readOnly={readOnly}
          css={{ width: 300, backgroundColor: theme.colors.backgroundPrimary }}
          placeholder={intl.formatMessage({
            defaultMessage: 'Select or type an assessment',
            description: 'Evaluation review > assessments > combobox placeholder',
          })}
          onKeyUp={(e) => {
            // Close menu on Enter if no item is highlighted. We need to use onKeyUp to avoid conflicts with Downshift
            if (comboboxState.highlightedIndex === -1 && e.key === 'Enter') {
              comboboxState.closeMenu();
            }
          }}
          comboboxState={comboboxState}
          formOnChange={(val) => {
            setFormValue(val);
          }}
        />
        {displayCombobox && (
          <TypeaheadComboboxMenu comboboxState={comboboxState} emptyText={addNewElementLabel} matchTriggerWidth>
            {visibleSuggestions.map((item, index) => (
              <TypeaheadComboboxMenuItem
                key={`${item.key}-${index}`}
                item={item}
                index={index}
                comboboxState={comboboxState}
                isDisabled={item?.disabled || false}
              >
                {item.label}
              </TypeaheadComboboxMenuItem>
            ))}
          </TypeaheadComboboxMenu>
        )}
      </TypeaheadComboboxRoot>
      <Spacer size="sm" />
      <div css={{ backgroundColor: theme.colors.backgroundPrimary, borderRadius: theme.general.borderRadiusBase }}>
        <Input.TextArea
          componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewassessmentupsertform.tsx_160"
          autoSize
          value={rationale ?? ''}
          onChange={(e) => setRationale(e.target.value)}
          placeholder={intl.formatMessage({
            defaultMessage: 'Add rationale (optional)',
            description: 'Evaluation review > assessments > rationale input placeholder',
          })}
        />
      </div>
      <Spacer size="sm" />
      <div css={{ display: 'flex', gap: theme.spacing.xs }}>
        <Button
          size="small"
          type="primary"
          componentId="mlflow.evaluations_review.confirm_edited_assessment_button"
          onClick={() => {
            // Assert form value
            if (!formValue) {
              return;
            }
            // Select assessment name either:
            // - from general suggestion when having multiple
            // - from already existing assessment when editing
            // - from custom value when provided
            const targetAssessmentName = formValue.rootAssessmentName ?? editedAssessment?.name ?? formValue.key;

            // Either use value or set it to "true" if we're using plain custom assessment name
            const value = targetAssessmentName !== formValue.key ? formValue.key : true;
            onSave({ value, rationale, assessmentName: targetAssessmentName });
          }}
          disabled={!formValue?.key}
        >
          <FormattedMessage
            defaultMessage="Confirm"
            description="Evaluation review > assessments > confirm assessment button label"
          />
        </Button>
        <Button
          size="small"
          type="tertiary"
          componentId="mlflow.evaluations_review.cancel_edited_assessment_button"
          onClick={onCancel}
        >
          <FormattedMessage
            defaultMessage="Cancel"
            description="Evaluation review > assessments > cancel assessment button label"
          />
        </Button>
      </div>
    </div>
  );
};
