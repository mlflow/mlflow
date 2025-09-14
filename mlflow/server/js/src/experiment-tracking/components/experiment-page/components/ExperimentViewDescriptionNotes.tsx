import type { ExperimentEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import {
  Button,
  ChevronDownIcon,
  ChevronUpIcon,
  Modal,
  PencilIcon,
  LegacyTooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { getExperimentTags } from '../../../reducers/Reducers';
import { NOTE_CONTENT_TAG } from '../../../utils/NoteUtils';
import type { ThunkDispatch } from '../../../../redux-types';
import React from 'react';
import 'react-mde/lib/styles/css/react-mde-all.css';
import ReactMde, { SvgIcon } from 'react-mde';
import {
  forceAnchorTagNewTab,
  getMarkdownConverter,
  sanitizeConvertedHtml,
} from '../../../../common/utils/MarkdownUtils';
import { FormattedMessage } from 'react-intl';
import { setExperimentTagApi } from '../../../actions';

const extractNoteFromTags = (tags: Record<string, KeyValueEntity>) =>
  Object.values(tags).find((t) => t.key === NOTE_CONTENT_TAG)?.value || undefined;

const toolbarCommands = [
  ['header', 'bold', 'italic', 'strikethrough'],
  ['link', 'code', 'image'],
  ['unordered-list', 'ordered-list'],
];

const converter = getMarkdownConverter();

const getSanitizedHtmlContent = (markdown: string | undefined) => {
  if (markdown) {
    const sanitized = sanitizeConvertedHtml(converter.makeHtml(markdown));
    return forceAnchorTagNewTab(sanitized);
  }
  return null;
};

export const ExperimentViewDescriptionNotes = ({
  experiment,
  editing,
  setEditing,
  setShowAddDescriptionButton,
  onNoteUpdated,
  defaultValue,
}: {
  experiment: ExperimentEntity;
  editing: boolean;
  setEditing: (editing: boolean) => void;
  setShowAddDescriptionButton: (show: boolean) => void;
  onNoteUpdated?: () => void;
  defaultValue?: string;
}) => {
  const storedNote = useSelector((state) => {
    const tags = getExperimentTags(experiment.experimentId, state);
    return tags ? extractNoteFromTags(tags) : '';
  });
  setShowAddDescriptionButton(!storedNote);

  const effectiveNote = storedNote || defaultValue;
  const [tmpNote, setTmpNote] = useState(effectiveNote);
  const [selectedTab, setSelectedTab] = useState<'write' | 'preview' | undefined>('write');
  const [isExpanded, setIsExpanded] = useState(false);

  const { theme } = useDesignSystemTheme();
  const PADDING_HORIZONTAL = 12;
  const DISPLAY_LINE_HEIGHT = 16;
  const COLLAPSE_MAX_HEIGHT = DISPLAY_LINE_HEIGHT + 2 * theme.spacing.sm;
  const MIN_EDITOR_HEIGHT = 200;
  const MAX_EDITOR_HEIGHT = 500;
  const MIN_PREVIEW_HEIGHT = 20;

  const dispatch = useDispatch<ThunkDispatch>();

  const handleSubmitEditNote = useCallback(
    (updatedNote?: string) => {
      setEditing(false);
      setShowAddDescriptionButton(!updatedNote);
      const action = setExperimentTagApi(experiment.experimentId, NOTE_CONTENT_TAG, updatedNote);
      dispatch(action).then(onNoteUpdated);
    },
    [experiment.experimentId, dispatch, setEditing, setShowAddDescriptionButton, onNoteUpdated],
  );

  return (
    <div>
      {effectiveNote && (
        <div
          style={{
            whiteSpace: isExpanded ? 'normal' : 'pre-wrap',
            lineHeight: theme.typography.lineHeightLg,
            background: theme.colors.backgroundSecondary,
            display: 'flex',
            alignItems: 'flex-start',
            padding: theme.spacing.xs,
          }}
        >
          <div
            style={{
              flexGrow: 1,
              marginRight: PADDING_HORIZONTAL,
              overflow: 'hidden',
              overflowWrap: isExpanded ? 'break-word' : undefined,
              padding: `${theme.spacing.sm}px ${PADDING_HORIZONTAL}px`,
              maxHeight: isExpanded ? 'none' : COLLAPSE_MAX_HEIGHT + 'px',
              wordBreak: 'break-word',
            }}
          >
            <div
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: getSanitizedHtmlContent(effectiveNote) }}
            />
          </div>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewdescriptionnotes.tsx_114"
            icon={<PencilIcon />}
            onClick={() => setEditing(true)}
            style={{ padding: `0px ${theme.spacing.sm}px` }}
          />
          {isExpanded ? (
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewdescriptionnotes.tsx_120"
              icon={<ChevronUpIcon />}
              onClick={() => setIsExpanded(false)}
              style={{ padding: `0px ${theme.spacing.sm}px` }}
            />
          ) : (
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewdescriptionnotes.tsx_126"
              icon={<ChevronDownIcon />}
              onClick={() => setIsExpanded(true)}
              style={{ padding: `0px ${theme.spacing.sm}px` }}
            />
          )}
        </div>
      )}
      <Modal
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewdescriptionnotes.tsx_141"
        title={
          <FormattedMessage
            defaultMessage="Add description"
            description="experiment page > description modal > title"
          />
        }
        visible={editing}
        okText={
          <FormattedMessage defaultMessage="Save" description="experiment page > description modal > save button" />
        }
        cancelText={
          <FormattedMessage defaultMessage="Cancel" description="experiment page > description modal > cancel button" />
        }
        onOk={() => {
          handleSubmitEditNote(tmpNote);
          setEditing(false);
        }}
        onCancel={() => {
          setTmpNote(effectiveNote);
          setEditing(false);
        }}
      >
        <React.Fragment>
          <ReactMde
            value={tmpNote}
            minEditorHeight={MIN_EDITOR_HEIGHT}
            maxEditorHeight={MAX_EDITOR_HEIGHT}
            minPreviewHeight={MIN_PREVIEW_HEIGHT}
            toolbarCommands={toolbarCommands}
            onChange={(value) => setTmpNote(value)}
            selectedTab={selectedTab}
            onTabChange={(newTab) => setSelectedTab(newTab)}
            generateMarkdownPreview={() => Promise.resolve(getSanitizedHtmlContent(tmpNote))}
            getIcon={(name) => (
              <LegacyTooltip title={name}>
                <span css={{ color: theme.colors.textPrimary }}>
                  <SvgIcon icon={name} />
                </span>
              </LegacyTooltip>
            )}
          />
        </React.Fragment>
      </Modal>
    </div>
  );
};
