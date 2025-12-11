import { useCallback, useState } from 'react';

import { useIntl } from '@databricks/i18n';

import { getAssessmentValueSuggestions } from '../components/GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, AssessmentDropdownSuggestionItem, RunEvaluationResultAssessment } from '../types';

/**
 * Manages the state of the edit assessment form. Provides methods to start adding and editing assessments.
 */
export const useEditAssessmentFormState = (
  assessmentHistory: RunEvaluationResultAssessment[],
  assessmentInfos?: AssessmentInfo[],
) => {
  const intl = useIntl();

  // The assessment that is currently being edited.
  const [editingAssessment, setEditingAssessment] = useState<RunEvaluationResultAssessment | undefined>(undefined);
  // True if the upsert form is currently being shown, false otherwise.
  const [showUpsertForm, setShowUpsertForm] = useState(false);
  // A list of suggestions for the value dropdown.
  const [suggestions, setSuggestions] = useState<AssessmentDropdownSuggestionItem[]>([]);

  const setFormState = useCallback(
    (isEditing: boolean, assessment?: RunEvaluationResultAssessment) => {
      setEditingAssessment(assessment);
      setShowUpsertForm(isEditing);
      setSuggestions(getAssessmentValueSuggestions(intl, assessment, assessmentHistory, assessmentInfos));
    },
    [intl, assessmentInfos, assessmentHistory],
  );
  const editAssessment = useCallback(
    (assessment: RunEvaluationResultAssessment) => setFormState(true, assessment),
    [setFormState],
  );
  const addAssessment = useCallback(() => setFormState(true, undefined), [setFormState]);
  const closeForm = useCallback(() => {
    setEditingAssessment(undefined);
    setShowUpsertForm(false);
  }, []);

  return {
    suggestions,
    editingAssessment,
    showUpsertForm,
    addAssessment,
    editAssessment,
    closeForm,
  };
};
