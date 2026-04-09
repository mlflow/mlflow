import { isNil } from 'lodash';
import { useCallback, useMemo } from 'react';

import { Checkbox, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { applyJsonPathToObject, isSpanInRange } from '../hooks/useTraceViewFiltering';
import type { PathSelection } from '../hooks/useTraceViews';
import { getTimelineTreeNodesList } from '../timeline-tree/TimelineTree.utils';

const hasSelection = (
  selections: PathSelection[] | undefined,
  spanId: string,
  path: string,
): boolean => !!selections?.some((s) => s.span_selector.span_id === spanId && s.path === path);

const toggleSelection = (
  selections: PathSelection[] | undefined,
  spanId: string,
  path: string,
): PathSelection[] => {
  const current = selections ?? [];
  const idx = current.findIndex((s) => s.span_selector.span_id === spanId && s.path === path);
  if (idx >= 0) {
    return current.filter((_, i) => i !== idx);
  }
  return [...current, { span_selector: { span_id: spanId }, path }];
};

export function ModelTraceExplorerDefaultSpanView({
  activeSpan,
  className,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  className?: string;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const { activeTraceView, editMode, topLevelNodes } = useModelTraceExplorerViewState();

  const firstRange = activeTraceView?.ranges?.[0];
  const filteredInputs = useMemo(
    () => applyJsonPathToObject(activeSpan?.inputs, firstRange?.input_path),
    [activeSpan?.inputs, firstRange?.input_path],
  );
  const filteredOutputs = useMemo(
    () => applyJsonPathToObject(activeSpan?.outputs, firstRange?.output_path),
    [activeSpan?.outputs, firstRange?.output_path],
  );

  const inputList = useMemo(() => createListFromObject(filteredInputs as any), [filteredInputs]);
  const outputList = useMemo(() => createListFromObject(filteredOutputs as any), [filteredOutputs]);

  // In edit mode, find which range the active span belongs to
  const editRangeIdx = useMemo(() => {
    if (!editMode.isEditMode || !editMode.draftView || !activeSpan) return null;
    const flatNodes = getTimelineTreeNodesList(topLevelNodes);
    for (let i = 0; i < editMode.draftView.ranges.length; i++) {
      if (isSpanInRange(activeSpan, flatNodes, editMode.draftView.ranges[i])) return i;
    }
    return null;
  }, [editMode.isEditMode, editMode.draftView, activeSpan, topLevelNodes]);

  const editRange = editRangeIdx !== null ? editMode.draftView?.ranges[editRangeIdx] : null;

  // Raw (unfiltered) input/output lists for edit mode checkboxes
  const rawInputList = useMemo(() => createListFromObject(activeSpan?.inputs as any), [activeSpan?.inputs]);
  const rawOutputList = useMemo(() => createListFromObject(activeSpan?.outputs as any), [activeSpan?.outputs]);

  const spanId = activeSpan ? String(activeSpan.key) : '';

  const handleToggleInput = useCallback(
    (key: string) => {
      if (editRangeIdx === null || !editRange) return;
      editMode.updateRange(editRangeIdx, {
        input_selections: toggleSelection(editRange.input_selections, spanId, `$.${key}`),
      });
    },
    [editMode, editRangeIdx, editRange, spanId],
  );

  const handleToggleOutput = useCallback(
    (key: string) => {
      if (editRangeIdx === null || !editRange) return;
      editMode.updateRange(editRangeIdx, {
        output_selections: toggleSelection(editRange.output_selections, spanId, `$.${key}`),
      });
    },
    [editMode, editRangeIdx, editRange, spanId],
  );

  if (isNil(activeSpan)) {
    return null;
  }

  const containsInputs = editRange ? rawInputList.length > 0 : inputList.length > 0;
  const containsOutputs = editRange ? rawOutputList.length > 0 : outputList.length > 0;

  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  const hasJsonPathFilter = !!(firstRange?.input_path || firstRange?.output_path);

  const displayInputList = editRange ? rawInputList : inputList;
  const displayOutputList = editRange ? rawOutputList : outputList;

  return (
    <div data-testid="model-trace-explorer-default-span-view">
      {!editRange && hasJsonPathFilter && activeTraceView && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            marginBottom: theme.spacing.sm,
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        >
          <Typography.Text size="sm" color="secondary">
            Filtered by: {activeTraceView.name}
          </Typography.Text>
        </div>
      )}
      {editRange && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            marginBottom: theme.spacing.sm,
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            backgroundColor: theme.colors.tagDefault,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        >
          <Typography.Text size="sm" color="secondary">
            Select fields to include in view for: {editRange.label}
          </Typography.Text>
        </div>
      )}
      {containsInputs && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          css={{ marginBottom: theme.spacing.sm }}
          sectionKey="input"
          title={
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
              }}
            >
              <FormattedMessage
                defaultMessage="Inputs"
                description="Model trace explorer > selected span > inputs header"
              />
            </div>
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {displayInputList.map(({ key, value }, index) => (
              <div key={key || index}>
                {editRange && editRangeIdx !== null && (
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: theme.spacing.xs,
                      marginBottom: theme.spacing.xs,
                    }}
                  >
                    <Checkbox
                      componentId={`edit-mode-input-selection.${spanId}.${key}`}
                      isChecked={hasSelection(editRange.input_selections, spanId, `$.${key}`)}
                      onChange={() => handleToggleInput(key)}
                      aria-label={`Include ${key} as input`}
                    />
                    <Typography.Text size="sm" color="secondary">
                      Include as input
                    </Typography.Text>
                  </div>
                )}
                <ModelTraceExplorerCodeSnippet
                  title={key}
                  data={value}
                  searchFilter={searchFilter}
                  activeMatch={activeMatch}
                  containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'inputs' && activeMatch.key === key}
                />
              </div>
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
      {containsOutputs && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          sectionKey="output"
          title={
            <div css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', width: '100%' }}>
              <FormattedMessage
                defaultMessage="Outputs"
                description="Model trace explorer > selected span > outputs header"
              />
            </div>
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {displayOutputList.map(({ key, value }, index) => (
              <div key={key || index}>
                {editRange && editRangeIdx !== null && (
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: theme.spacing.xs,
                      marginBottom: theme.spacing.xs,
                    }}
                  >
                    <Checkbox
                      componentId={`edit-mode-output-selection.${spanId}.${key}`}
                      isChecked={hasSelection(editRange.output_selections, spanId, `$.${key}`)}
                      onChange={() => handleToggleOutput(key)}
                      aria-label={`Include ${key} as output`}
                    />
                    <Typography.Text size="sm" color="secondary">
                      Include as output
                    </Typography.Text>
                  </div>
                )}
                <ModelTraceExplorerCodeSnippet
                  title={key}
                  data={value}
                  searchFilter={searchFilter}
                  activeMatch={activeMatch}
                  containsActiveMatch={
                    isActiveMatchSpan && activeMatch.section === 'outputs' && activeMatch.key === key
                  }
                />
              </div>
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
    </div>
  );
}
