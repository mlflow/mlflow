import { isNil, isString } from 'lodash';
import { useMemo } from 'react';

import {
  PencilIcon,
  SparkleDoubleIcon,
  UserIcon,
  useDesignSystemTheme,
  Button,
  CheckCircleIcon,
  XCircleIcon,
  WarningIcon,
  XCircleFillIcon,
  HoverCard,
  Typography,
  InfoSmallIcon,
  BracketsXIcon,
  DangerIcon,
} from '@databricks/design-system';
import type { ThemeType } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import {
  KnownEvaluationResultAssessmentValueLabel,
  KnownEvaluationResultAssessmentValueMapping,
  getEvaluationResultAssessmentValue,
  hasBeenEditedByHuman,
  KnownEvaluationResultAssessmentStringValue,
  KnownEvaluationResultAssessmentValueMissingTooltip,
  ASSESSMENTS_DOC_LINKS,
  getJudgeMetricsLink,
} from './GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, RunEvaluationResultAssessment } from '../types';
import {
  getEvaluationResultAssessmentBackgroundColor,
  getEvaluationResultIconColor,
  getEvaluationResultTextColor,
} from '../utils/Colors';
import { displayFloat } from '../utils/DisplayUtils';
import { ASSESSMENT_RATIONAL_HOVER_DETAILS_VIEW } from '../utils/EvaluationLogging';
import { useMarkdownConverter } from '../utils/MarkdownUtils';

export const isAssessmentPassing = (
  assessmentInfo: AssessmentInfo,
  assessmentValue?: string | number | boolean | null,
) => {
  if (!isNil(assessmentValue)) {
    if (assessmentInfo.dtype === 'pass-fail') {
      if (assessmentValue === KnownEvaluationResultAssessmentStringValue.YES) {
        return true;
      } else if (assessmentValue === KnownEvaluationResultAssessmentStringValue.NO) {
        return false;
      }
    } else if (assessmentInfo.dtype === 'boolean') {
      return assessmentValue === true;
    }
  }
  return undefined;
};

function getAssessmentTagDisplayValue(
  theme: ThemeType,
  intl: IntlShape,
  type: 'value' | 'assessment-value',
  assessmentInfo: AssessmentInfo,
  editedByHuman: boolean,
  assessment?: RunEvaluationResultAssessment,
  isRootCauseAssessment?: boolean,
): { tagText: JSX.Element | string; icon: JSX.Element; fullTagText?: JSX.Element | string | undefined } {
  let tagText: string | JSX.Element = '';
  let fullTagText: string | JSX.Element | undefined = undefined;
  let icon: JSX.Element = <></>;

  const errorDisplayValue = {
    tagText: (
      <FormattedMessage defaultMessage="Error" description="Error assessment status in the evaluations table." />
    ),
    icon: <DangerIcon css={{ color: theme.colors.textValidationWarning }} />,
  };

  const nullDisplayValue = {
    tagText: (
      <span css={{ fontStyle: 'italic' }}>
        <FormattedMessage defaultMessage="null" description="Null value in the evaluations table." />
      </span>
    ),
    icon: <WarningIcon css={{ color: theme.colors.grey400 }} />,
  };

  const value = assessment ? getEvaluationResultAssessmentValue(assessment) : undefined;
  const isError = Boolean(assessment?.errorMessage);

  if (isError) {
    return errorDisplayValue;
  }

  if (assessmentInfo.dtype === 'pass-fail' || assessmentInfo.dtype === 'boolean') {
    const isPassing = isAssessmentPassing(assessmentInfo, value);
    let displayValueText = '';
    if (assessmentInfo.dtype === 'pass-fail') {
      // Known assessments are all pass / fail.
      if (isPassing === true) {
        displayValueText = intl.formatMessage({
          defaultMessage: 'Pass',
          description: 'Passing evaluation status in the evaluations table.',
        });
      } else if (isPassing === false) {
        displayValueText = intl.formatMessage({
          defaultMessage: 'Fail',
          description: 'Failing evaluation status in the evaluations table.',
        });
      } else {
        return nullDisplayValue;
      }
    } else if (isPassing === true) {
      displayValueText = intl.formatMessage({
        defaultMessage: 'True',
        description: 'True value in the evaluations table.',
      });
    } else if (isPassing === false) {
      displayValueText = intl.formatMessage({
        defaultMessage: 'False',
        description: 'False value in the evaluations table.',
      });
    } else {
      return nullDisplayValue;
    }

    const iconColor = getEvaluationResultIconColor(theme, assessmentInfo, assessment);
    icon =
      isPassing === true ? (
        <CheckCircleIcon
          css={{
            color: iconColor,
          }}
        />
      ) : isPassing === false ? (
        isRootCauseAssessment ? (
          <XCircleFillIcon
            css={{
              color: iconColor,
            }}
          />
        ) : (
          <XCircleIcon
            css={{
              color: iconColor,
            }}
          />
        )
      ) : (
        <WarningIcon
          css={{
            color: iconColor,
          }}
        />
      );

    if (type === 'assessment-value') {
      const knownMapping = KnownEvaluationResultAssessmentValueMapping[assessmentInfo.name];

      if (knownMapping) {
        const messageDescriptor = value
          ? knownMapping[value.toString()] ?? knownMapping[KnownEvaluationResultAssessmentStringValue.YES]
          : knownMapping[KnownEvaluationResultAssessmentStringValue.YES];
        if (messageDescriptor) {
          tagText = <FormattedMessage {...messageDescriptor} values={{ value }} />;
        }
      } else {
        tagText = (
          <>
            {assessmentInfo.displayName}: {value}
          </>
        );
      }
    } else if (type === 'value') {
      if (isNil(isPassing)) {
        tagText = <span css={{ fontStyle: 'italic' }}>{displayValueText}</span>;
      } else {
        tagText = displayValueText;
      }
    }
  } else if (assessmentInfo.dtype === 'numeric') {
    const roundedValue = !isNil(value) ? displayFloat(value as number | undefined | null) : value;

    if (type === 'assessment-value') {
      tagText = (
        <>
          {assessmentInfo.displayName}: {roundedValue}
        </>
      );
      fullTagText = (
        <>
          {assessmentInfo.displayName}: {value}
        </>
      );
    } else {
      if (isNil(roundedValue)) {
        return nullDisplayValue;
      }
      tagText = `${roundedValue}`;
      fullTagText = `${value}`;
    }
  } else {
    // Wrap nulls in italics.
    if (isNil(value)) {
      return nullDisplayValue;
    }
    const valueElement = <>{String(value)}</>;
    if (type === 'assessment-value') {
      if (isNil(value)) {
        tagText = <>{assessmentInfo.displayName}</>;
      } else {
        tagText = (
          <>
            {assessmentInfo.displayName}: {valueElement}
          </>
        );
      }
    } else {
      tagText = valueElement;
    }
  }
  return { tagText, icon, fullTagText };
}

export const EvaluationsReviewAssessmentTag = ({
  assessment,
  onEdit,
  active = false,
  disableJudgeTypeIcon,
  showRationaleInTooltip = false,
  showPassFailText = false,
  hideAssessmentName = false,
  iconOnly = false,
  disableTooltip = false,
  isRootCauseAssessment,
  assessmentInfo,
  type,
  count,
}: {
  assessment?: RunEvaluationResultAssessment;
  onEdit?: () => void;
  active?: boolean;
  disableJudgeTypeIcon?: boolean;
  showRationaleInTooltip?: boolean;
  showPassFailText?: boolean;
  hideAssessmentName?: boolean;
  iconOnly?: boolean;
  disableTooltip?: boolean;
  isRootCauseAssessment?: boolean;
  assessmentInfo: AssessmentInfo;
  type: 'value' | 'assessment-value';
  count?: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const value = assessment ? getEvaluationResultAssessmentValue(assessment) : undefined;
  const isPassing = useMemo(() => isAssessmentPassing(assessmentInfo, value), [value, assessmentInfo]);

  const iconColor = getEvaluationResultIconColor(theme, assessmentInfo, assessment);
  const textColor = getEvaluationResultTextColor(theme, assessmentInfo, assessment);

  let errorMessage: string | undefined = undefined;
  if (
    assessment?.errorMessage ||
    (isPassing === undefined && assessment && KnownEvaluationResultAssessmentValueMissingTooltip[assessment.name])
  ) {
    errorMessage =
      assessment.errorMessage ||
      intl.formatMessage(KnownEvaluationResultAssessmentValueMissingTooltip[assessment.name]);
  }

  const knownValueLabel = assessment ? KnownEvaluationResultAssessmentValueLabel[assessment.name] : undefined;
  const assessmentTitle = useMemo(
    () => (knownValueLabel ? intl.formatMessage(knownValueLabel) : assessment?.name),
    [assessment, knownValueLabel, intl],
  );
  const learnMoreLink = useMemo(
    () => (assessment ? getJudgeMetricsLink(ASSESSMENTS_DOC_LINKS[assessment.name]) : undefined),
    [assessment],
  );

  const { makeHTML } = useMarkdownConverter();

  const rationaleHTML = useMemo(() => {
    const rationale = assessment?.rationale;
    return isString(rationale) ? makeHTML(rationale) : undefined;
  }, [assessment, makeHTML]);

  const editedByHuman = useMemo(() => !isNil(assessment) && hasBeenEditedByHuman(assessment), [assessment]);

  const { tagText, icon, fullTagText } = useMemo(
    () =>
      getAssessmentTagDisplayValue(theme, intl, type, assessmentInfo, editedByHuman, assessment, isRootCauseAssessment),
    [theme, intl, type, assessmentInfo, assessment, isRootCauseAssessment, editedByHuman],
  );

  const tagContent = (
    <>
      {tagText}
      {count && count > 1 ? ` (${count})` : ''}
    </>
  );

  // Hide human assessment tags when not defined.
  const hideTag = assessmentInfo.source?.sourceType === 'HUMAN' && !assessment?.source?.sourceId;
  if (hideTag) {
    return <></>;
  }

  const tagElement = (
    <div>
      <EvaluationsReviewTag
        iconOnly={iconOnly}
        hideAssessmentName={hideAssessmentName}
        tagContent={tagContent}
        active={active}
        icon={icon}
        iconColor={iconColor}
        textColor={textColor}
        sourceIcon={
          assessmentInfo.isCustomMetric ? (
            <BracketsXIcon />
          ) : assessment && editedByHuman ? (
            <UserIcon />
          ) : (
            <SparkleDoubleIcon />
          )
        }
        backgroundColor={getEvaluationResultAssessmentBackgroundColor(theme, assessmentInfo, assessment)}
        disableSourceTypeIcon={disableJudgeTypeIcon && !editedByHuman}
        hasBeenEditedByHuman={editedByHuman}
        onEdit={onEdit}
      />
    </div>
  );

  return (
    <>
      {disableTooltip ? (
        tagElement
      ) : (
        <HoverCard
          side="bottom"
          content={
            <div
              css={{
                maxWidth: '25rem',
                display: 'flex',
                flexDirection: 'column',
                overflowWrap: 'break-word',
                wordBreak: 'break-word',
                gap: theme.spacing.sm,
              }}
            >
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.xs,
                }}
              >
                <div
                  css={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <div
                    css={{
                      display: 'flex',
                      gap: theme.spacing.sm,
                      alignItems: 'center',
                    }}
                  >
                    <Typography.Title
                      css={{
                        marginBottom: 0,
                      }}
                    >
                      {assessmentTitle}
                    </Typography.Title>
                    <EvaluationsReviewTag
                      iconOnly={iconOnly}
                      hideAssessmentName={hideAssessmentName}
                      tagContent={fullTagText ? fullTagText : tagContent}
                      active={active}
                      icon={icon}
                      iconColor={iconColor}
                      textColor={textColor}
                      sourceIcon={
                        assessmentInfo.isCustomMetric ? (
                          <BracketsXIcon />
                        ) : assessment && hasBeenEditedByHuman(assessment) ? (
                          <UserIcon />
                        ) : (
                          <SparkleDoubleIcon />
                        )
                      }
                      backgroundColor={getEvaluationResultAssessmentBackgroundColor(theme, assessmentInfo, assessment)}
                      disableSourceTypeIcon={disableJudgeTypeIcon}
                      hasBeenEditedByHuman={editedByHuman}
                    />
                  </div>
                  {learnMoreLink && (
                    <a
                      href={learnMoreLink}
                      target="_blank"
                      rel="noreferrer"
                      css={{
                        height: '16px',
                      }}
                    >
                      <InfoSmallIcon />
                    </a>
                  )}
                </div>
                {isRootCauseAssessment && (
                  <Typography.Hint>
                    {intl.formatMessage({
                      defaultMessage: 'This assessment is the root cause of the overall evaluation failure.',
                      description:
                        'Root cause assessment hint that explains that this assessment is causing the overall assessment to fail.',
                    })}
                  </Typography.Hint>
                )}
              </div>
              {showRationaleInTooltip && assessment && rationaleHTML && (
                <div>
                  <>
                    <span
                      css={{
                        display: 'contents',
                        '& p': {
                          marginBottom: 0,
                        },
                      }}
                      // eslint-disable-next-line react/no-danger
                      dangerouslySetInnerHTML={{ __html: rationaleHTML }}
                    />
                  </>
                </div>
              )}
              {errorMessage && (
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.xs,
                  }}
                >
                  <span
                    css={{
                      color: theme.colors.textValidationWarning,
                      fontStyle: 'italic',
                    }}
                  >
                    {errorMessage}
                  </span>
                </div>
              )}
            </div>
          }
          trigger={tagElement}
        />
      )}
    </>
  );
};

const EvaluationsReviewTag = ({
  iconOnly,
  hideAssessmentName,
  tagContent,
  active,
  sourceIcon,
  icon,
  iconColor,
  textColor,
  backgroundColor,
  disableSourceTypeIcon,
  hasBeenEditedByHuman,
  onEdit,
}: {
  iconOnly: boolean;
  hideAssessmentName: boolean;
  tagContent: string | number | true | JSX.Element | undefined;
  active?: boolean;
  sourceIcon?: JSX.Element;
  icon: JSX.Element;
  iconColor: string;
  textColor: string;
  backgroundColor: string;
  disableSourceTypeIcon?: boolean;
  hasBeenEditedByHuman?: boolean;
  onEdit?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const svgSize = iconOnly ? 18 : 12;

  return (
    <div
      css={{
        // TODO: Use <Badge /> when it's aligned with design
        display: 'inline-flex',
        height: iconOnly ? svgSize : 20,
        width: iconOnly ? svgSize : hideAssessmentName ? 'fit-content' : '',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: iconOnly ? '0' : onEdit ? '0 0 0 8px' : '0 8px',
        gap: theme.spacing.sm,
        borderRadius: iconOnly ? '50%' : theme.legacyBorders.borderRadiusMd,
        backgroundColor: backgroundColor,
        boxShadow: `inset 0 0 1px 1px ${active ? theme.colors.borderAccessible : 'transparent'}`,
        // border: iconOnly ? '' : `1px solid ${getEvaluationBorderColor(theme, assessment)}`,
        fontSize: theme.typography.fontSizeSm,
        svg: { width: svgSize, height: svgSize },
        whiteSpace: 'nowrap',
      }}
    >
      {icon}
      {tagContent && (
        <span
          css={{
            color: textColor,
          }}
        >
          {tagContent}
        </span>
      )}
      {disableSourceTypeIcon !== true ? (
        <span
          css={{
            color: iconColor,
          }}
        >
          {sourceIcon}
        </span>
      ) : (
        <></>
      )}
      {onEdit && (
        <Button
          componentId="mlflow.evaluations_review.edit_assessment_button"
          onClick={onEdit}
          size="small"
          icon={
            <PencilIcon
              css={{
                ':hover': {
                  color: theme.colors.actionDefaultTextHover,
                },
              }}
            />
          }
        />
      )}
    </div>
  );
};
