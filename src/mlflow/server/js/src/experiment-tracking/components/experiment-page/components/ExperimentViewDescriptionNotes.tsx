import { ExperimentEntity, KeyValueEntity } from '../../../types';
import {
  Button,
  ChevronDownIcon,
  ChevronUpIcon,
  Modal,
  PencilIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { getExperimentTags } from '../../../reducers/Reducers';
import { NOTE_CONTENT_TAG } from '../../../utils/NoteUtils';
import { useFetchExperiments } from '../hooks/useFetchExperiments';
import { ThunkDispatch } from '../../../../redux-types';
import React from 'react';
import ReactMde, { SvgIcon } from 'react-mde';
import { forceAnchorTagNewTab, getConverter, sanitizeConvertedHtml } from '../../../../common/utils/MarkdownUtils';
import { FormattedMessage } from 'react-intl';

const extractNoteFromTags = (tags: Record<string, KeyValueEntity>) =>
  Object.values(tags).find((t) => t.getKey() === NOTE_CONTENT_TAG)?.value || undefined;

const toolbarCommands = [
  ['header', 'bold', 'italic', 'strikethrough'],
  ['link', 'code', 'image'],
  ['unordered-list', 'ordered-list'],
];

const converter = getConverter();

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
}: {
  experiment: ExperimentEntity;
  editing: boolean;
  setEditing: (editing: boolean) => void;
  setShowAddDescriptionButton: (show: boolean) => void;
}) => {
  const storedNote = useSelector((state) => {
    const tags = getExperimentTags(experiment.experiment_id, state);
    return tags ? extractNoteFromTags(tags) : '';
  });
  setShowAddDescriptionButton(!storedNote);

  const [tmpNote, setTmpNote] = useState(storedNote);
  const [selectedTab, setSelectedTab] = useState<'write' | 'preview' | undefined>('write');
  const [isExpanded, setIsExpanded] = useState(false);

  const { theme } = useDesignSystemTheme();
  const PADDING_HORIZONTAL = 12;
  const DISPLAY_LINE_HEIGHT = 16;
  const COLLAPSE_MAX_HEIGHT = DISPLAY_LINE_HEIGHT + 2 * theme.spacing.sm;
  const MIN_EDITOR_HEIGHT = 200;
  const MAX_EDITOR_HEIGHT = 500;
  const MIN_PREVIEW_HEIGHT = 20;

  const {
    actions: { setExperimentTagApi },
  } = useFetchExperiments();

  const dispatch = useDispatch<ThunkDispatch>();

  const handleSubmitEditNote = useCallback(
    (updatedNote: any) => {
      setEditing(false);
      setShowAddDescriptionButton(!updatedNote);
      const action = setExperimentTagApi(experiment.experiment_id, NOTE_CONTENT_TAG, updatedNote);
      dispatch(action);
    },
    [experiment.experiment_id, setExperimentTagApi, dispatch, setEditing, setShowAddDescriptionButton],
  );

  return (
    <div>
      {tmpNote && (
        <div
          style={{
            whiteSpace: isExpanded ? 'normal' : 'pre',
            lineHeight: theme.typography.lineHeightSm,
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
            }}
          >
            <div
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: getSanitizedHtmlContent(tmpNote) }}
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
          setTmpNote(storedNote);
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
              <Tooltip title={name}>
                <span css={{ color: theme.colors.textPrimary }}>
                  <SvgIcon icon={name} />
                </span>
              </Tooltip>
            )}
          />
        </React.Fragment>
      </Modal>
    </div>
  );
};
