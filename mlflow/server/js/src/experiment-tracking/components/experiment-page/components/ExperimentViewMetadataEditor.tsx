import type { ExperimentEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import {
  Alert,
  Button,
  ChevronDownIcon,
  ChevronUpIcon,
  FormUI,
  Input,
  Modal,
  PencilIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { getExperiments, getExperimentTags } from '../../../reducers/Reducers';
import { NOTE_CONTENT_TAG } from '../../../utils/NoteUtils';
import type { ThunkDispatch } from '../../../../redux-types';
import { SvgIcon } from 'react-mde';
import {
  forceAnchorTagNewTab,
  getMarkdownConverter,
  sanitizeConvertedHtml,
} from '../../../../common/utils/MarkdownUtils';
import { ThemeAwareReactMde } from '../../../../common/components/EditableNote';
import { FormattedMessage, useIntl } from 'react-intl';
import { getExperimentApi, setExperimentTagApi, updateExperimentApi } from '../../../actions';
import { getExperimentNameValidator } from '../../../../common/forms/validations';
import { useInvalidateExperimentList } from '../hooks/useExperimentListQuery';
import { canModifyExperiment, canRenameExperiment } from '../utils/experimentPage.common-utils';

const extractNoteFromTags = (tags: Record<string, KeyValueEntity>) =>
  Object.values(tags).find((t) => t.key === NOTE_CONTENT_TAG)?.value;

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

export const ExperimentViewMetadataEditor = ({
  experiment,
  editing,
  setEditing,
  onNoteUpdated,
  defaultValue,
}: {
  experiment: ExperimentEntity;
  editing: boolean;
  setEditing: (editing: boolean) => void;
  onNoteUpdated?: () => void;
  defaultValue?: string;
}) => {
  const storedNote = useSelector((state) => {
    const tags = getExperimentTags(experiment.experimentId, state);
    return tags ? extractNoteFromTags(tags) : '';
  });
  const existingExperimentNames = useSelector((state) =>
    getExperiments(state)
      .map((exp) => exp.name)
      .filter((name) => name !== experiment.name),
  );

  const effectiveNote = storedNote ?? defaultValue;
  const [tmpName, setTmpName] = useState(experiment.name);
  const [tmpNote, setTmpNote] = useState(effectiveNote);
  const [nameError, setNameError] = useState<string | undefined>();
  const [saveError, setSaveError] = useState<string | undefined>();
  const [isSaving, setIsSaving] = useState(false);
  const [selectedTab, setSelectedTab] = useState<'write' | 'preview'>('write');
  const [isExpanded, setIsExpanded] = useState(false);

  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const PADDING_HORIZONTAL = 12;
  const DISPLAY_LINE_HEIGHT = 16;
  const COLLAPSE_MAX_HEIGHT = DISPLAY_LINE_HEIGHT + 2 * theme.spacing.sm;
  const MIN_EDITOR_HEIGHT = 200;
  const MAX_EDITOR_HEIGHT = 500;
  const MIN_PREVIEW_HEIGHT = 20;

  const dispatch = useDispatch<ThunkDispatch>();
  const invalidateExperimentList = useInvalidateExperimentList();
  const canEditMetadata = canModifyExperiment(experiment);
  const canRename = canRenameExperiment(experiment);

  const validateExperimentName = useCallback(
    async (experimentName: string) =>
      new Promise<string | undefined>((resolve) => {
        getExperimentNameValidator(() => existingExperimentNames)(undefined, experimentName, resolve);
      }),
    [existingExperimentNames],
  );

  const handleSubmitEditExperiment = useCallback(
    async (updatedName?: string, updatedNote?: string) => {
      const trimmedName = updatedName?.trim() ?? '';
      const currentNote = effectiveNote ?? '';
      const updatedNoteValue = updatedNote ?? '';
      const hasNoteChanged = canEditMetadata && updatedNoteValue !== currentNote;
      const shouldRename = canRename && trimmedName !== experiment.name;
      if (canRename && !trimmedName) {
        setNameError(
          intl.formatMessage({
            defaultMessage: 'Please input a new name for the experiment.',
            description: 'experiment page > edit experiment modal > empty experiment name validation error',
          }),
        );
        return;
      }

      if (shouldRename) {
        const validationError = await validateExperimentName(trimmedName);
        if (validationError) {
          setNameError(validationError);
          return;
        }
      }

      setNameError(undefined);
      setSaveError(undefined);
      setIsSaving(true);

      try {
        if (shouldRename) {
          await dispatch(updateExperimentApi(experiment.experimentId, trimmedName));
          invalidateExperimentList();
        }

        if (hasNoteChanged) {
          await dispatch(setExperimentTagApi(experiment.experimentId, NOTE_CONTENT_TAG, updatedNote));
        }

        if (shouldRename) {
          await dispatch(getExperimentApi(experiment.experimentId)).catch(() => undefined);
        }

        onNoteUpdated?.();
        setEditing(false);
      } catch (e: any) {
        setSaveError(
          e?.message ||
            intl.formatMessage({
              defaultMessage: 'Failed to update experiment. Please try again.',
              description: 'experiment page > edit experiment modal > generic save error',
            }),
        );
      } finally {
        setIsSaving(false);
      }
    },
    [
      dispatch,
      experiment.experimentId,
      experiment.name,
      effectiveNote,
      invalidateExperimentList,
      intl,
      onNoteUpdated,
      setEditing,
      validateExperimentName,
      canEditMetadata,
      canRename,
    ],
  );

  const sanitizedContent = getSanitizedHtmlContent(effectiveNote);
  const hasContent = sanitizedContent && sanitizedContent.trim().length > 0;
  const getIcon = useCallback(
    (name: string) => {
      return (
        <Tooltip componentId="mlflow.experiment-tracking.experiment-description.edit" content={name}>
          <span css={{ color: theme.colors.textPrimary }}>
            <SvgIcon icon={name} />
          </span>
        </Tooltip>
      );
    },
    [theme],
  );

  useEffect(() => {
    if (editing) {
      setTmpName(experiment.name);
      setTmpNote(effectiveNote);
      setNameError(undefined);
      setSaveError(undefined);
    }
  }, [editing, effectiveNote, experiment.name]);

  return (
    <div
      css={{
        paddingBottom: theme.spacing.sm,
        borderBottom: `1px solid ${theme.colors.border}`,
      }}
    >
      {hasContent && (
        <div
          css={{
            background: theme.colors.backgroundSecondary,
            display: 'flex',
            alignItems: 'flex-start',
            padding: theme.spacing.xs,
          }}
        >
          <div
            css={{
              flexGrow: 1,
              marginRight: PADDING_HORIZONTAL,
              overflow: 'hidden',
              overflowWrap: isExpanded ? 'break-word' : undefined,
              padding: `${theme.spacing.sm}px ${PADDING_HORIZONTAL}px`,
              maxHeight: isExpanded ? 'none' : COLLAPSE_MAX_HEIGHT + 'px',
              wordBreak: 'break-word',
              '&>:first-child': {
                marginBlockStart: 0,
                ...(isExpanded
                  ? {}
                  : {
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                    }),
              },
            }}
          >
            <div
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: sanitizedContent }}
            />
          </div>
          {canEditMetadata && (
            <Button
              componentId="mlflow.experiment.metadata_editor.edit_button"
              data-testid="experiment-metadata-editor-edit-button"
              icon={<PencilIcon />}
              onClick={() => setEditing(true)}
              style={{ padding: `0px ${theme.spacing.sm}px` }}
            />
          )}
          {isExpanded ? (
            <Button
              componentId="mlflow.experiment.metadata_editor.collapse_button"
              icon={<ChevronUpIcon />}
              onClick={() => setIsExpanded(false)}
              style={{ padding: `0px ${theme.spacing.sm}px` }}
            />
          ) : (
            <Button
              componentId="mlflow.experiment.metadata_editor.expand_button"
              icon={<ChevronDownIcon />}
              onClick={() => setIsExpanded(true)}
              style={{ padding: `0px ${theme.spacing.sm}px` }}
            />
          )}
        </div>
      )}
      <Modal
        componentId="mlflow.experiment.metadata_editor.modal"
        title={
          <FormattedMessage
            defaultMessage="Edit experiment"
            description="experiment page > edit experiment modal > title"
          />
        }
        visible={editing}
        okButtonProps={{ loading: isSaving }}
        okText={
          <FormattedMessage defaultMessage="Save" description="experiment page > edit experiment modal > save button" />
        }
        cancelText={
          <FormattedMessage
            defaultMessage="Cancel"
            description="experiment page > edit experiment modal > cancel button"
          />
        }
        onOk={() => handleSubmitEditExperiment(tmpName, tmpNote)}
        onCancel={() => {
          setTmpName(experiment.name);
          setTmpNote(effectiveNote);
          setNameError(undefined);
          setSaveError(undefined);
          setEditing(false);
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {saveError && (
            <Alert
              componentId="mlflow.experiment.metadata_editor.error"
              closable={false}
              message={saveError}
              type="error"
            />
          )}
          {canRename && (
            <div>
              <FormUI.Label htmlFor="mlflow.experiment.edit.name">
                <FormattedMessage
                  defaultMessage="Experiment name"
                  description="experiment page > edit experiment modal > experiment name label"
                />
              </FormUI.Label>
              <Input
                componentId="mlflow.experiment.edit.name"
                id="mlflow.experiment.edit.name"
                value={tmpName}
                onChange={(e) => {
                  setTmpName(e.target.value);
                  if (nameError) {
                    setNameError(undefined);
                  }
                }}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Enter experiment name',
                  description: 'experiment page > edit experiment modal > experiment name placeholder',
                })}
                autoFocus
                validationState={nameError ? 'error' : undefined}
              />
              {nameError && <FormUI.Message type="error" message={nameError} />}
            </div>
          )}
          {canEditMetadata && (
            <div>
              <FormUI.Label htmlFor="mlflow.experiment.edit.description">
                <FormattedMessage
                  defaultMessage="Description"
                  description="experiment page > edit experiment modal > description label"
                />
              </FormUI.Label>
              <ThemeAwareReactMde
                value={tmpNote || ''}
                minEditorHeight={MIN_EDITOR_HEIGHT}
                maxEditorHeight={MAX_EDITOR_HEIGHT}
                minPreviewHeight={MIN_PREVIEW_HEIGHT}
                toolbarCommands={toolbarCommands}
                onChange={(value) => setTmpNote(value)}
                selectedTab={selectedTab}
                onTabChange={(newTab) => setSelectedTab(newTab)}
                generateMarkdownPreview={() => Promise.resolve(getSanitizedHtmlContent(tmpNote))}
                getIcon={getIcon}
              />
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
};
