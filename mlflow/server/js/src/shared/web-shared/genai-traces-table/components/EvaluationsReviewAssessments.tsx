import { first, isString } from 'lodash';
import { useEffect, useMemo, useState } from 'react';

import {
  AssistantIcon,
  BugIcon,
  Button,
  ChevronRightIcon,
  ChevronUpIcon,
  PlusIcon,
  Spacer,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { EvaluationsReviewAssessmentDetailedHistory } from './EvaluationsReviewAssessmentDetailedHistory';
import { EvaluationsReviewAssessmentTag } from './EvaluationsReviewAssessmentTag';
import { EvaluationsReviewAssessmentTooltip } from './EvaluationsReviewAssessmentTooltip';
import { EvaluationsReviewAssessmentUpsertForm } from './EvaluationsReviewAssessmentUpsertForm';
import {
  createDraftEvaluationResultAssessmentObject,
  getEvaluationResultAssessmentValue,
  isAssessmentMissing,
  isDraftAssessment,
  KnownEvaluationResultAssessmentName,
  KnownEvaluationResultAssessmentValueLabel,
} from './GenAiEvaluationTracesReview.utils';
import { useEditAssessmentFormState } from '../hooks/useEditAssessmentFormState';
import type { AssessmentInfo, RunEvaluationResultAssessmentDraft, RunEvaluationResultAssessment } from '../types';
import { EXPANDED_ASSESSMENT_DETAILS_VIEW } from '../utils/EvaluationLogging';
import { useMarkdownConverter } from '../utils/MarkdownUtils';

/**
 * Displays an expanded assessment with rationale and edit history.
 * Expanded assessments each has its own edit form.
 */
const ExpandedAssessment = ({
  assessmentsType, // Type of the assessments, e.g. 'overall', 'response', 'retrieval'. Used for component IDs.
  assessmentName, // Name of the assessment.
  assessmentHistory, // A list of assessment history.
  rootCauseAssessment, // The root cause assessment causing this to fail.
  onUpsertAssessment, // Callback to upsert an assessment. This is called when the user saves an assessment. Any pre-saving logic should be done here.
  allowEditing = false, // Whether editing is allowed.
  options, // A list of known assessment names as options for the dropdown.
  inputs, // Dependency array to control the refresh of the states.
  assessmentInfo,
  assessmentInfos,
}: {
  assessmentsType: 'overall' | 'response' | 'retrieval';
  assessmentName: string;
  assessmentHistory: RunEvaluationResultAssessment[];
  rootCauseAssessment?: RunEvaluationResultAssessment;
  onUpsertAssessment: (assessment: RunEvaluationResultAssessmentDraft) => void;
  allowEditing?: boolean;
  options?: KnownEvaluationResultAssessmentName[];
  inputs?: any;
  assessmentInfo: AssessmentInfo;
  assessmentInfos: AssessmentInfo[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const isOverallAssessment = assessmentsType === 'overall';

  const {
    suggestions,
    editingAssessment,
    showUpsertForm: isEditing,
    editAssessment,
    closeForm,
  } = useEditAssessmentFormState(assessmentHistory, assessmentInfos);

  // Clear the states when the inputs change
  useEffect(() => {
    // Close the form if it's open
    closeForm();
  }, [inputs, closeForm]);

  const assessment = first(assessmentHistory);

  const intlLabel = KnownEvaluationResultAssessmentValueLabel[assessmentName];
  const label = intlLabel ? intl.formatMessage(intlLabel) : assessmentName;

  const hasValue = Boolean(assessment && getEvaluationResultAssessmentValue(assessment));
  const isDraft = Boolean(assessment && isDraftAssessment(assessment));

  const isEditable = allowEditing && (hasValue || isDraft);

  const { makeHTML } = useMarkdownConverter();

  const suggestedActionHtml = useMemo(() => {
    const suggestedAction = assessment?.rootCauseAssessment?.suggestedActions;
    return isString(suggestedAction) ? makeHTML(suggestedAction) : null;
  }, [assessment, makeHTML]);

  return (
    <div
      key={assessmentName}
      css={{
        display: 'block',
        marginBottom: !isOverallAssessment ? theme.spacing.md : undefined,
      }}
    >
      {!isEditing && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          {assessmentName !== KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT && (
            <div
              css={{
                display: 'flex',
                gap: theme.spacing.sm,
                alignItems: 'center',
              }}
            >
              <EvaluationsReviewAssessmentTag
                assessment={assessment}
                aria-label={label}
                disableJudgeTypeIcon={isAssessmentMissing(assessment)}
                onEdit={
                  isEditable
                    ? () => {
                        const assessmentToEdit = first(assessmentHistory);
                        assessmentToEdit && editAssessment(assessmentToEdit);
                      }
                    : undefined
                }
                assessmentInfo={assessmentInfo}
                type="assessment-value"
              />
            </div>
          )}
          <EvaluationsReviewAssessmentDetailedHistory
            history={assessmentHistory}
            alwaysExpanded={isOverallAssessment}
          />

          {rootCauseAssessment && (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                marginTop: theme.spacing.xs,
              }}
            >
              {/* Root cause failure */}
              <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                <BugIcon color="danger" />
                <Typography.Text bold>
                  <FormattedMessage
                    defaultMessage="Root cause failure:"
                    description="Evaluation review > assessments > root cause failure > title"
                  />
                </Typography.Text>
                <EvaluationsReviewAssessmentTag
                  assessment={rootCauseAssessment}
                  isRootCauseAssessment
                  aria-label={label}
                  assessmentInfo={assessmentInfo}
                  type="assessment-value"
                />
              </div>
              <EvaluationsReviewAssessmentDetailedHistory
                history={[rootCauseAssessment]}
                alwaysExpanded={isOverallAssessment}
              />
              {suggestedActionHtml && (
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.sm,
                  }}
                >
                  <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                    <AssistantIcon color="ai" />
                    <Typography.Text bold>
                      <FormattedMessage
                        defaultMessage="Suggested actions"
                        description="Evaluation review > assessments > suggested actions > title"
                      />
                    </Typography.Text>
                  </div>
                  {/* eslint-disable-next-line react/no-danger */}
                  <span css={{ display: 'contents' }} dangerouslySetInnerHTML={{ __html: suggestedActionHtml }} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
      {isEditing && (
        <>
          <EvaluationsReviewAssessmentUpsertForm
            key={editingAssessment?.name}
            editedAssessment={editingAssessment}
            valueSuggestions={suggestions}
            onCancel={closeForm}
            onSave={({ value, rationale, assessmentName }) => {
              const defaultAssessmentName = isOverallAssessment
                ? KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT
                : '';
              const assessment = createDraftEvaluationResultAssessmentObject({
                name: assessmentName ?? editingAssessment?.name ?? defaultAssessmentName,
                isOverallAssessment: isOverallAssessment,
                value,
                rationale,
              });
              onUpsertAssessment(assessment);
              closeForm();
            }}
          />
        </>
      )}
    </div>
  );
};

/**
 * Displays a list of assessments in expanded mode along with an add assessment button at the end.
 */
const ExpandedAssessments = ({
  assessmentsType, // Type of the assessments, e.g. 'overall', 'response', 'retrieval'. Used for component IDs.
  assessmentsByName, // A list of assessments by name.
  rootCauseAssessment, // The root cause assessment causing this to fail.
  onUpsertAssessment, // Callback to upsert an assessment. This is called when the user saves an assessment. Any pre-saving logic should be done here.
  allowEditing = false, // Whether editing is allowed.
  allowMoreThanOne = false, // Whether allow more than one assessment.
  options, // A list of known assessment names as options for the dropdown.
  inputs, // Dependency array to control the refresh of the states.
  assessmentInfos,
}: {
  assessmentsType: 'overall' | 'response' | 'retrieval';
  assessmentsByName: [string, RunEvaluationResultAssessment[]][];
  rootCauseAssessment?: RunEvaluationResultAssessment;
  onUpsertAssessment: (assessment: RunEvaluationResultAssessmentDraft) => void;
  allowEditing?: boolean;
  allowMoreThanOne?: boolean;
  options?: KnownEvaluationResultAssessmentName[];
  inputs?: any;
  assessmentInfos: AssessmentInfo[];
}) => {
  const { theme } = useDesignSystemTheme();

  const isOverallAssessment = assessmentsType === 'overall';

  const nonEmptyAssessments = assessmentsByName.filter(([_, assessmentList]) => assessmentList.length > 0);

  const { suggestions, editingAssessment, showUpsertForm, addAssessment, closeForm } = useEditAssessmentFormState(
    nonEmptyAssessments.flatMap(([_key, assessmentList]) => assessmentList),
    assessmentInfos,
  );

  // Clear the states when the inputs change
  useEffect(() => {
    // Close the form if it's open
    closeForm();
  }, [inputs, closeForm]);

  const containsAssessments = Object.keys(nonEmptyAssessments).length > 0;
  const showAddAssessmentButton = allowEditing && (allowMoreThanOne || !containsAssessments);

  return (
    <>
      <div
        // comment for copybara formatting
        css={{ display: 'block', flexWrap: 'wrap', gap: theme.spacing.xs }}
      >
        {nonEmptyAssessments.map(([key, assessmentList]) => {
          const assessmentInfo = assessmentInfos.find((info) => info.name === key);
          if (!assessmentInfo) {
            return <div css={{ display: 'none' }} key={key} />;
          }
          return (
            <ExpandedAssessment
              key={key}
              assessmentsType={assessmentsType}
              assessmentName={key}
              assessmentHistory={assessmentList}
              rootCauseAssessment={rootCauseAssessment}
              onUpsertAssessment={onUpsertAssessment}
              allowEditing={allowEditing}
              options={options}
              inputs={inputs}
              assessmentInfo={assessmentInfo}
              assessmentInfos={assessmentInfos}
            />
          );
        })}
        {showAddAssessmentButton && (
          <Button
            componentId={`mlflow.evaluations_review.add_assessment_${assessmentsType}_button`}
            onClick={addAssessment}
            icon={<PlusIcon />}
            size="small"
          >
            <FormattedMessage
              defaultMessage="Add assessment"
              description="Evaluation review > assessments > add assessment button label"
            />
          </Button>
        )}
      </div>
      {showUpsertForm && (
        <>
          <Spacer size="sm" />
          <EvaluationsReviewAssessmentUpsertForm
            valueSuggestions={suggestions}
            onCancel={closeForm}
            onSave={({ value, rationale, assessmentName }) => {
              const defaultAssessmentName = isOverallAssessment
                ? KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT
                : assessmentName || '';
              const assessment = createDraftEvaluationResultAssessmentObject({
                name: assessmentName ?? editingAssessment?.name ?? defaultAssessmentName,
                isOverallAssessment: isOverallAssessment,
                value,
                rationale,
              });
              onUpsertAssessment(assessment);
              closeForm();
            }}
          />
        </>
      )}
    </>
  );
};

/**
 * Displays a list of assessments in compact mode.
 * Compact assessments share the same edit form.
 */
const CompactAssessments = ({
  assessmentsType, // Type of the assessments, e.g. 'overall', 'response', 'retrieval'. Used for component IDs.
  assessmentsByName, // A list of assessments by name.
  onUpsertAssessment, // Callback to upsert an assessment. This is called when the user saves an assessment. Any pre-saving logic should be done here.
  allowEditing = false, // Whether editing is allowed.
  allowMoreThanOne = false, // Whether allow more than one assessment.
  options, // A list of known assessment names as options for the dropdown.
  inputs, // Dependency array to control the refresh of the states.
  assessmentInfos,
}: {
  assessmentsType: 'overall' | 'response' | 'retrieval';
  assessmentsByName: [string, RunEvaluationResultAssessment[]][];
  onUpsertAssessment: (assessment: RunEvaluationResultAssessmentDraft) => void;
  allowEditing?: boolean;
  allowMoreThanOne?: boolean;
  options?: KnownEvaluationResultAssessmentName[];
  inputs?: any;
  assessmentInfos: AssessmentInfo[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const isOverallAssessment = assessmentsType === 'overall';

  const { suggestions, editingAssessment, showUpsertForm, addAssessment, editAssessment, closeForm } =
    useEditAssessmentFormState(
      assessmentsByName.flatMap(([_key, assessmentList]) => assessmentList),
      assessmentInfos,
    );

  // Clear the states when the inputs change
  useEffect(() => {
    // Close the form if it's open
    closeForm();
  }, [inputs, closeForm]);

  const nonEmptyAssessments = assessmentsByName.filter(([_, assessmentList]) => assessmentList.length > 0);

  const containsAssessments = Object.keys(nonEmptyAssessments).length > 0;
  const showAddAssessmentButton = allowEditing && (allowMoreThanOne || !containsAssessments);

  return (
    <>
      <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs, alignItems: 'center' }}>
        {nonEmptyAssessments.map(([key, assessmentList]) => {
          const assessment = first(assessmentList);
          const assessmentInfo = assessmentInfos.find((info) => info.name === key);

          if (!assessmentInfo) {
            // eslint-disable-next-line react/jsx-key -- TODO(FEINF-1756)
            return <></>;
          }

          const intlLabel = KnownEvaluationResultAssessmentValueLabel[key];
          const label = intlLabel ? intl.formatMessage(intlLabel) : key;

          const hasValue = Boolean(assessment && getEvaluationResultAssessmentValue(assessment));
          const isDraft = Boolean(assessment && isDraftAssessment(assessment));

          const isEditable = allowEditing && (hasValue || isDraft) && assessmentInfo.isEditable;

          return (
            <div
              key={key}
              css={{
                display: 'contents',
              }}
            >
              <EvaluationsReviewAssessmentTooltip assessmentHistory={assessmentList}>
                <EvaluationsReviewAssessmentTag
                  assessment={assessment}
                  aria-label={label}
                  active={editingAssessment?.name === key && showUpsertForm}
                  disableJudgeTypeIcon={isAssessmentMissing(assessment)}
                  onEdit={
                    isEditable
                      ? () => {
                          const assessmentToEdit = first(assessmentList);
                          assessmentToEdit && editAssessment(assessmentToEdit);
                        }
                      : undefined
                  }
                  assessmentInfo={assessmentInfo}
                  type="assessment-value"
                />
              </EvaluationsReviewAssessmentTooltip>
            </div>
          );
        })}
        {showAddAssessmentButton && (
          <Button
            componentId={`mlflow.evaluations_review.add_assessment_${assessmentsType}_button`}
            onClick={addAssessment}
            icon={<PlusIcon />}
            size="small"
          >
            <FormattedMessage
              defaultMessage="Add assessment"
              description="Evaluation review > assessments > add assessment button label"
            />
          </Button>
        )}
      </div>
      {showUpsertForm && (
        <>
          <Spacer size="sm" />
          <EvaluationsReviewAssessmentUpsertForm
            key={editingAssessment?.name}
            editedAssessment={editingAssessment}
            valueSuggestions={suggestions}
            onCancel={closeForm}
            onSave={({ value, rationale, assessmentName }) => {
              const defaultAssessmentName = isOverallAssessment
                ? KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT
                : assessmentName || '';
              const assessment = createDraftEvaluationResultAssessmentObject({
                name: assessmentName ?? editingAssessment?.name ?? defaultAssessmentName,
                isOverallAssessment: isOverallAssessment,
                value,
                rationale,
              });
              onUpsertAssessment(assessment);
              closeForm();
            }}
          />
        </>
      )}
    </>
  );
};

/**
 * Displays a list of assessments with an option to expand and see the detailed view.
 */
export const EvaluationsReviewAssessments = ({
  assessmentsType, // Type of the assessments, e.g. 'overall', 'response', 'retrieval'. Used for component IDs.
  assessmentsByName, // A list of assessments by name.
  rootCauseAssessment, // The root cause assessment causing this to fail.
  onUpsertAssessment, // Callback to upsert an assessment. This is called when the user saves an assessment. Any pre-saving logic should be done here.
  allowEditing = false, // Whether editing is allowed.
  allowMoreThanOne = false, // Whether allow more than one assessment.
  alwaysExpanded = false, // Whether the detailed view is always expanded.
  options, // A list of known assessment names as options for the dropdown.
  inputs, // Dependency array to control the refresh of the states.
  assessmentInfos,
}: {
  assessmentsType: 'overall' | 'response' | 'retrieval';
  assessmentsByName: [string, RunEvaluationResultAssessment[]][];
  rootCauseAssessment?: RunEvaluationResultAssessment;
  onUpsertAssessment: (assessment: RunEvaluationResultAssessmentDraft) => void;
  allowEditing?: boolean;
  allowMoreThanOne?: boolean;
  alwaysExpanded?: boolean;
  options?: KnownEvaluationResultAssessmentName[];
  inputs?: any;
  assessmentInfos: AssessmentInfo[];
}) => {
  // True if in expanded view, false otherwise.
  const [isExpandedView, setIsExpandedView] = useState(false);
  const showExpandedView = alwaysExpanded || isExpandedView;

  const nonEmptyAssessments = assessmentsByName.filter(([_, assessmentList]) => assessmentList.length > 0);

  // Remove the overall assessment if it's not an overall assessment and we're not showing overall assessments.
  let filteredAssessmentsByName = assessmentsByName;
  if (assessmentsType !== 'overall') {
    filteredAssessmentsByName = assessmentsByName.filter(
      ([key, _]) => key !== KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
    );
  }

  const containsAssessments = Object.keys(nonEmptyAssessments).length > 0;

  return (
    <>
      {showExpandedView && (
        <ExpandedAssessments
          assessmentsType={assessmentsType}
          assessmentsByName={filteredAssessmentsByName}
          rootCauseAssessment={rootCauseAssessment}
          onUpsertAssessment={onUpsertAssessment}
          allowEditing={allowEditing}
          allowMoreThanOne={allowMoreThanOne}
          options={options}
          inputs={inputs}
          assessmentInfos={assessmentInfos}
        />
      )}
      {!showExpandedView && (
        <CompactAssessments
          assessmentsType={assessmentsType}
          assessmentsByName={filteredAssessmentsByName}
          onUpsertAssessment={onUpsertAssessment}
          allowEditing={allowEditing}
          allowMoreThanOne={allowMoreThanOne}
          options={options}
          inputs={inputs}
          assessmentInfos={assessmentInfos}
        />
      )}

      {containsAssessments && !alwaysExpanded && (
        <>
          <Spacer size="sm" />
          <Button
            size="small"
            type="tertiary"
            componentId={`mlflow.evaluations_review.see_assessment_details_${assessmentsType}_button`}
            icon={showExpandedView ? <ChevronUpIcon /> : <ChevronRightIcon />}
            onClick={() => setIsExpandedView((mode) => !mode)}
          >
            {showExpandedView ? (
              <FormattedMessage
                defaultMessage="Hide details"
                description="Evaluation review > assessments > hide details button"
              />
            ) : (
              <FormattedMessage
                defaultMessage="See details"
                description="Evaluation review > assessments > see details button"
              />
            )}
          </Button>
        </>
      )}
    </>
  );
};
