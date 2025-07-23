import { isNil } from 'lodash';
import { useState } from 'react';

import { useDesignSystemTheme, Typography, Button, PlusIcon, Tooltip, DangerIcon } from '@databricks/design-system';

import { AssessmentCreateForm } from './AssessmentCreateForm';
import { getAssessmentDisplayName } from './AssessmentsPane.utils';
import { FeedbackValueGroup } from './FeedbackValueGroup';
import type { FeedbackAssessment } from '../ModelTrace.types';

export const FeedbackGroup = ({
  name,
  valuesMap,
  traceId,
  activeSpanId,
}: {
  name: string;
  valuesMap: { [value: string]: FeedbackAssessment[] };
  traceId: string;
  activeSpanId?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const displayName = getAssessmentDisplayName(name);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const hasError = Object.values(valuesMap)
    .flat()
    .some((feedback) => !isNil(feedback.feedback.error));

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        marginBottom: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.sm + theme.spacing.xs,
        paddingTop: theme.spacing.sm,
        gap: theme.spacing.sm,
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flex: 1, minWidth: 0 }}>
          <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap' }}>
            {displayName}
          </Typography.Text>
          {hasError && <DangerIcon css={{ flexShrink: 0 }} color="danger" />}
        </div>
        <Tooltip content="Add new feedback" componentId="shared.model-trace-explorer.add-feedback-in-group-tooltip">
          <Button
            componentId="shared.model-trace-explorer.add-feedback"
            css={{ flexShrink: 0, marginRight: -theme.spacing.xs }}
            size="small"
            icon={<PlusIcon />}
            onClick={() => setShowCreateForm(true)}
          />
        </Tooltip>
      </div>
      {Object.entries(valuesMap).map(([jsonValue, feedbacks]) => (
        <FeedbackValueGroup jsonValue={jsonValue} feedbacks={feedbacks} key={jsonValue} />
      ))}
      {showCreateForm && (
        <AssessmentCreateForm
          assessmentName={name}
          spanId={activeSpanId}
          traceId={traceId}
          setExpanded={setShowCreateForm}
        />
      )}
    </div>
  );
};
