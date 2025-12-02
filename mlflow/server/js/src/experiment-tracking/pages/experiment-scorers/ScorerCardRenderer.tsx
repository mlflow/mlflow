import React from 'react';
import type { Control, UseFormSetValue, UseFormGetValues } from 'react-hook-form';
import {
  useDesignSystemTheme,
  Typography,
  Tag,
  Button,
  Card,
  ChevronDownIcon,
  ChevronRightIcon,
  CircleIcon,
  PencilIcon,
  OverflowIcon,
  DropdownMenu,
  TrashIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { isNil } from 'lodash';
import type { ScheduledScorer } from './types';
import { getTypeDisplayName, getTypeIcon, getTypeColor, getStatusTag } from './scorerCardUtils';
import LLMScorerFormRenderer, { type LLMScorerFormData } from './LLMScorerFormRenderer';
import CustomCodeScorerFormRenderer, { type CustomCodeScorerFormData } from './CustomCodeScorerFormRenderer';
import { COMPONENT_ID_PREFIX, SCORER_FORM_MODE } from './constants';

interface ScorerCardOverflowMenuProps {
  onDelete: () => void;
}

const ScorerCardOverflowMenu: React.FC<ScorerCardOverflowMenuProps> = ({ onDelete }) => {
  const handleClick = React.useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onDelete();
    },
    [onDelete],
  );

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button componentId={`${COMPONENT_ID_PREFIX}.overflow-button`} size="small" icon={<OverflowIcon />} />
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        <DropdownMenu.Item componentId={`${COMPONENT_ID_PREFIX}.delete-button`} onClick={handleClick}>
          <DropdownMenu.IconWrapper>
            <TrashIcon />
          </DropdownMenu.IconWrapper>
          <FormattedMessage defaultMessage="Delete" description="Delete judge button" />
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

interface ScorerCardRendererProps {
  scorer: ScheduledScorer;
  isExpanded: boolean;
  onCardClick: () => void;
  onExpandToggle: (e: React.MouseEvent) => void;
  onEditClick: (e: React.MouseEvent) => void;
  onDeleteClick: () => void;
  control: Control<LLMScorerFormData | CustomCodeScorerFormData>;
  setValue: UseFormSetValue<LLMScorerFormData | CustomCodeScorerFormData>;
  getValues: UseFormGetValues<LLMScorerFormData | CustomCodeScorerFormData>;
}

const ScorerCardRenderer: React.FC<ScorerCardRendererProps> = ({
  scorer,
  isExpanded,
  onCardClick,
  onExpandToggle,
  onEditClick,
  onDeleteClick,
  control,
  setValue,
  getValues,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <Card
      componentId={`${COMPONENT_ID_PREFIX}.scorer-card`}
      css={{
        padding: theme.spacing.md,
        position: 'relative',
        width: '100%',
        boxSizing: 'border-box',
        cursor: 'pointer',
      }}
      onClick={onCardClick}
    >
      {/* Header with title, expand button and action buttons */}
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr auto',
          gap: theme.spacing.xs,
          alignItems: 'flex-start',
        }}
      >
        <Button
          componentId={`${COMPONENT_ID_PREFIX}.expand-button`}
          icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          size="small"
          type="tertiary"
          onClick={onExpandToggle}
          css={{
            padding: theme.spacing.xs,
          }}
          disabled={false}
        />
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
            <Typography.Title level={4} css={{ margin: 0, marginBottom: '0 !important' }}>
              {scorer.name}
            </Typography.Title>
            <Tag
              componentId={`${COMPONENT_ID_PREFIX}.scorer-type-tag`}
              color={getTypeColor(scorer)}
              icon={getTypeIcon(scorer)}
            >
              {getTypeDisplayName(scorer, intl)}
            </Tag>
          </div>
          {/* Metadata such as sample rate, filter string and version - only show when collapsed */}
          {!isExpanded && (!isNil(scorer.sampleRate) || scorer.filterString || !isNil(scorer.version)) ? (
            <div
              css={{
                display: 'flex',
                gap: theme.spacing.sm,
                alignItems: 'center',
              }}
            >
              {!scorer.disableMonitoring && !isNil(scorer.sampleRate) && (
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <Typography.Hint>
                    <FormattedMessage defaultMessage="Sample rate:" description="Sample rate label for scorer" />
                  </Typography.Hint>
                  <Typography.Hint>
                    <FormattedMessage
                      defaultMessage="{sampleRatePercent}%"
                      description="Sample rate value for scorer"
                      values={{ sampleRatePercent: scorer.sampleRate }}
                    />
                  </Typography.Hint>
                </div>
              )}
              {!scorer.disableMonitoring && !isNil(scorer.sampleRate) && scorer.filterString && (
                <CircleIcon css={{ color: theme.colors.textSecondary, fontSize: '6px' }} />
              )}
              {!scorer.disableMonitoring && scorer.filterString && (
                <Typography.Hint>
                  <FormattedMessage
                    defaultMessage="Filter: {filterString}"
                    description="Filter display for scorer"
                    values={{ filterString: scorer.filterString }}
                  />
                </Typography.Hint>
              )}
              {!isNil(scorer.version) && (
                <Typography.Hint>
                  <FormattedMessage
                    defaultMessage="Version {version}"
                    description="Version display for judge"
                    values={{ version: scorer.version }}
                  />
                </Typography.Hint>
              )}
            </div>
          ) : null}
        </div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          {!scorer.disableMonitoring && (
            <Tag
              componentId={`${COMPONENT_ID_PREFIX}.scorer-status-tag`}
              color={getStatusTag(scorer, intl).color}
              icon={getStatusTag(scorer, intl).icon}
            >
              {getStatusTag(scorer, intl).text}
            </Tag>
          )}
          <Button
            componentId={`${COMPONENT_ID_PREFIX}.edit-button`}
            size="small"
            icon={<PencilIcon />}
            onClick={onEditClick}
          >
            <FormattedMessage defaultMessage="Edit" description="Edit button for judge" />
          </Button>
          <ScorerCardOverflowMenu onDelete={onDeleteClick} />
        </div>
      </div>
      {/* Expanded content - aligned with scorer name, display mode only */}
      {isExpanded && (
        <div
          css={{
            gridColumn: '2 / -1',
            marginTop: theme.spacing.md,
            cursor: 'auto',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {scorer.type === 'llm' && (
            <LLMScorerFormRenderer
              mode={SCORER_FORM_MODE.DISPLAY}
              control={control as Control<LLMScorerFormData>}
              setValue={setValue as UseFormSetValue<LLMScorerFormData>}
              getValues={getValues as UseFormGetValues<LLMScorerFormData>}
            />
          )}
          {scorer.type === 'custom-code' && (
            <CustomCodeScorerFormRenderer
              control={control as Control<CustomCodeScorerFormData>}
              mode={SCORER_FORM_MODE.DISPLAY}
            />
          )}
        </div>
      )}
    </Card>
  );
};

export default ScorerCardRenderer;
