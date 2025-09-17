import {
  Button,
  DangerModal,
  DragIcon,
  DropdownMenu,
  Input,
  Modal,
  OverflowIcon,
  PencilIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ChartSectionConfig } from '../../../../types';
import { RunsChartsAddChartMenu } from '../RunsChartsAddChartMenu';
import type { RunsChartType } from '../../runs-charts.types';
import { useEffect, useRef, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useDragAndDropElement } from '@mlflow/mlflow/src/common/hooks/useDragAndDropElement';
import { CheckIcon } from '@databricks/design-system';
import { METRIC_CHART_SECTION_HEADER_SIZE } from '../../../MetricChartsAccordion';
import cx from 'classnames';

export interface RunsChartsSectionHeaderProps {
  index: number;
  section: ChartSectionConfig;
  sectionChartsLength: number;
  addNewChartCard: (metricSectionId: string) => (type: RunsChartType) => void;
  onDeleteSection: (sectionId: string) => void;
  onAddSection: (sectionId: string, above: boolean) => void;
  editSection: number;
  onSetEditSection: React.Dispatch<React.SetStateAction<number>>;
  onSetSectionName: (sectionId: string, name: string) => void;
  onSectionReorder: (sourceSectionId: string, targetSectionId: string) => void;
  isExpanded: boolean;
  supportedChartTypes?: RunsChartType[] | undefined;
  /**
   * Set to "true" to hide various controls (e.g. edit, add, delete) in the section header.
   */
  hideExtraControls: boolean;
}

export const RunsChartsSectionHeader = ({
  index,
  section,
  sectionChartsLength,
  addNewChartCard,
  onDeleteSection,
  onAddSection,
  editSection,
  onSetEditSection,
  onSetSectionName,
  onSectionReorder,
  isExpanded,
  hideExtraControls,
  supportedChartTypes,
}: RunsChartsSectionHeaderProps) => {
  const { theme } = useDesignSystemTheme();
  // Change name locally for better performance
  const [tmpSectionName, setTmpSectionName] = useState(section.name);
  // State to check if element is being dragged
  const [isDraggingHandle, setIsDraggingHandle] = useState(false);

  // Ref and state to measure the width of the section name
  const sectionNameRef = useRef<HTMLDivElement>(null);
  const confirmButtonRef = useRef<HTMLButtonElement>(null);
  const [sectionNameWidth, setSectionNameWidth] = useState(0.0);

  // Delete section modal
  const [isDeleteSectionModalOpen, setIsDeleteSectionModalOpen] = useState(false);

  const stopPropagation = (e: any) => {
    e.stopPropagation();
  };

  const deleteModalConfirm = () => {
    onDeleteSection(section.uuid);
  };

  const deleteModalCancel = () => {
    setIsDeleteSectionModalOpen(false);
  };

  const deleteSection = () => {
    setIsDeleteSectionModalOpen(true);
  };

  const addSectionAbove = () => {
    onAddSection(section.uuid, true);
  };

  const addSectionBelow = () => {
    onAddSection(section.uuid, false);
  };

  const onEdit = (e: React.MouseEvent<HTMLElement>) => {
    e.stopPropagation();
    onSetEditSection(index);
  };

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTmpSectionName(e.target.value);
  };

  const onSubmit = (e: React.KeyboardEvent<HTMLInputElement> | React.MouseEvent<HTMLElement>) => {
    e.stopPropagation();
    if (!tmpSectionName.trim()) {
      e.preventDefault();
      return;
    }
    onSetEditSection(-1);
    onSetSectionName(section.uuid, tmpSectionName);
  };

  const onEsc = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') {
      e.stopPropagation();
      onSetEditSection(-1);
      setTmpSectionName(section.name);
    }
  };

  const onBlur = (e: React.FocusEvent) => {
    if (e.relatedTarget === confirmButtonRef.current) {
      return;
    }
    onSetEditSection(-1);
    onSetSectionName(section.uuid, tmpSectionName);
  };

  useEffect(() => {
    if (!sectionNameRef.current) {
      return;
    }

    const resizeObserver = new ResizeObserver(([entry]) => {
      setSectionNameWidth(entry.contentRect.width);
    });

    resizeObserver.observe(sectionNameRef.current);

    return () => resizeObserver.disconnect();
  }, []);

  // For explicitness:
  const EDITABLE_LABEL_PADDING_WIDTH = 6;
  const EDITABLE_LABEL_BORDER_WIDTH = 1;
  const EDITABLE_LABEL_OFFSET = EDITABLE_LABEL_PADDING_WIDTH + EDITABLE_LABEL_BORDER_WIDTH;

  const isCurrentlyEdited = editSection === index;
  const [isCurrentlyHovered, setIsCurrentlyHovered] = useState(false);

  const { dragHandleRef, dragPreviewRef, dropTargetRef, isOver, isDragging } = useDragAndDropElement({
    dragGroupKey: 'sections',
    dragKey: section.uuid,
    onDrop: onSectionReorder,
  });

  return (
    <>
      <div
        role="figure"
        css={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          padding: `${theme.spacing.xs}px 0px`,
          height: `${METRIC_CHART_SECTION_HEADER_SIZE}px`,
          '.section-element-visibility-on-hover': {
            visibility: isCurrentlyHovered ? 'visible' : 'hidden',
            opacity: isCurrentlyHovered ? 1 : 0,
          },
          '.section-element-visibility-on-hover-and-not-drag': {
            visibility: isCurrentlyHovered ? 'visible' : 'hidden',
            opacity: isCurrentlyHovered ? (isDraggingHandle ? 0 : 1) : 0,
          },
          '.section-element-hidden-on-edit': { display: isCurrentlyEdited ? 'none' : 'inherit' },
        }}
        onMouseMove={() => setIsCurrentlyHovered(true)}
        onMouseLeave={() => setIsCurrentlyHovered(false)}
        ref={(element) => {
          // Use this element for both drag preview and drop target
          dropTargetRef?.(element);
          dragPreviewRef?.(element);
        }}
        data-testid="experiment-view-compare-runs-section-header"
      >
        {isOver && (
          // Visual overlay for target drop element
          <div
            css={{
              position: 'absolute',
              inset: 0,
              backgroundColor: theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100,
              border: `2px dashed ${theme.colors.blue400}`,
              opacity: 0.75,
            }}
          />
        )}
        <div
          style={{
            maxWidth: '40%',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <div
            ref={sectionNameRef}
            style={{
              position: !isCurrentlyEdited ? 'relative' : 'absolute',
              visibility: !isCurrentlyEdited ? 'visible' : 'hidden',
              textOverflow: isCurrentlyEdited ? undefined : 'ellipsis',
              maxWidth: '100%',
              overflow: 'clip',
              paddingLeft: EDITABLE_LABEL_OFFSET,
              whiteSpace: 'pre',
            }}
          >
            {tmpSectionName}
          </div>
          {editSection === index && (
            <Input
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_220"
              autoFocus
              onClick={stopPropagation}
              onMouseDown={stopPropagation}
              onMouseUp={stopPropagation}
              onDoubleClick={stopPropagation}
              onChange={onChange}
              value={tmpSectionName}
              css={{
                color: theme.colors.textPrimary,
                fontWeight: 600,
                padding: `1px ${EDITABLE_LABEL_PADDING_WIDTH}px 1px ${EDITABLE_LABEL_PADDING_WIDTH}px`,
                background: theme.colors.backgroundSecondary,
                minWidth: '50px',
                width: sectionNameWidth + 2 * EDITABLE_LABEL_OFFSET,
                position: 'relative',
                lineHeight: theme.typography.lineHeightBase,
                maxWidth: '100%',
              }}
              onKeyDown={onEsc}
              onPressEnter={onSubmit}
              dangerouslyAppendEmotionCSS={{ '&&': { minHeight: '20px !important' } }}
              onBlur={onBlur}
            />
          )}
          <div
            css={{
              padding: theme.spacing.xs,
              position: 'relative',
            }}
            style={{
              visibility: !isCurrentlyEdited ? 'visible' : 'hidden',
              display: isCurrentlyEdited ? 'none' : 'inherit',
            }}
          >
            {`(${sectionChartsLength})`}
          </div>
          {!hideExtraControls && (
            <div className="section-element-visibility-on-hover-and-not-drag section-element-hidden-on-edit">
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_sections_runscomparesectionheader.tsx_246"
                onClick={onEdit}
                aria-label="Icon label"
                icon={<PencilIcon />}
              />
            </div>
          )}
        </div>
        {editSection === index && !hideExtraControls && (
          <div style={{ padding: `0 ${theme.spacing.xs}px` }}>
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_sections_runscomparesectionheader.tsx_251"
              onClick={onSubmit}
              icon={<CheckIcon />}
              ref={confirmButtonRef}
            />
          </div>
        )}
        {!hideExtraControls && (
          <div
            className="section-element-visibility-on-hover section-element-hidden-on-edit"
            css={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', cursor: 'grab' }}
          >
            <DragIcon
              rotate={90}
              style={{ color: theme.colors.textSecondary }}
              ref={dragHandleRef}
              onMouseDown={() => setIsDraggingHandle(true)}
              onMouseLeave={() => {
                setIsDraggingHandle(false);
              }}
              data-testid="experiment-view-compare-runs-section-header-drag-handle"
            />
          </div>
        )}
        {!hideExtraControls && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              right: '0',
              transform: 'translate(0, -50%)',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <div
              onClick={stopPropagation}
              onMouseDown={stopPropagation}
              onMouseUp={stopPropagation}
              onDoubleClick={stopPropagation}
              className="section-element-visibility-on-hover-and-not-drag section-element-hidden-on-edit"
            >
              <DropdownMenu.Root modal={false}>
                <DropdownMenu.Trigger asChild>
                  <Button
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_sections_runscomparesectionheader.tsx_288"
                    icon={<OverflowIcon />}
                  />
                </DropdownMenu.Trigger>
                <DropdownMenu.Content>
                  <DropdownMenu.Item
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_321"
                    onClick={addSectionAbove}
                  >
                    <FormattedMessage
                      defaultMessage="Add section above"
                      description="Experiment page > compare runs > chart section > add section above label"
                    />
                  </DropdownMenu.Item>
                  <DropdownMenu.Item
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_327"
                    onClick={addSectionBelow}
                  >
                    <FormattedMessage
                      defaultMessage="Add section below"
                      description="Experiment page > compare runs > chart section > add section below label"
                    />
                  </DropdownMenu.Item>
                  <DropdownMenu.Item
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_333"
                    onClick={deleteSection}
                  >
                    <FormattedMessage
                      defaultMessage="Delete section"
                      description="Experiment page > compare runs > chart section > delete section label"
                    />
                  </DropdownMenu.Item>
                </DropdownMenu.Content>
              </DropdownMenu.Root>
              <DangerModal
                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_351"
                visible={isDeleteSectionModalOpen}
                onOk={deleteModalConfirm}
                onCancel={deleteModalCancel}
                title="Delete section"
              >
                <FormattedMessage
                  defaultMessage="Deleting the section will permanently remove it and the charts it contains. This cannot be undone."
                  description="Experiment page > compare runs > chart section > delete section warning message"
                />
              </DangerModal>
            </div>

            <div
              onClick={stopPropagation}
              onMouseDown={stopPropagation}
              onMouseUp={stopPropagation}
              onDoubleClick={stopPropagation}
              className={cx(
                {
                  'section-element-visibility-on-hover-and-not-drag': !isExpanded,
                },
                'section-element-hidden-on-edit',
              )}
              css={{
                alignSelf: 'flex-end',
                marginLeft: theme.spacing.xs,
              }}
            >
              <RunsChartsAddChartMenu
                onAddChart={addNewChartCard(section.uuid)}
                supportedChartTypes={supportedChartTypes}
              />
            </div>
          </div>
        )}
      </div>
    </>
  );
};
