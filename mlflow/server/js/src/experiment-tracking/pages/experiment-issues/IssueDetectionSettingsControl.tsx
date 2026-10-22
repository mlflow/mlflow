import {
  Alert,
  Button,
  CloseIcon,
  FormUI,
  GearIcon,
  Input,
  Modal,
  PlusIcon,
  ToggleButton,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useIntl } from 'react-intl';
import {
  ALL_ISSUE_CATEGORIES,
  ISSUE_CATEGORY_DEFINITIONS,
} from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import {
  defaultIssueDetectionSettings,
  noFocusAreaSelected,
  type IssueDetectionSettings,
} from './issueDetectionSettings';
import { useIssueDetectionSettings } from './useIssueDetectionSettings';

const COMPONENT_ID = 'mlflow.experiment-issues.detection-settings';

/** Kept in step with the server's own cap, so the form cannot save text the run would truncate. */
const MAX_INSTRUCTIONS_LENGTH = 5000;
const MAX_CATEGORY_LENGTH = 80;

const isBuiltInCategory = (category: string) => ALL_ISSUE_CATEGORIES.some((id) => id === category);

export interface IssueDetectionSettingsControlProps {
  experimentId: string;
}

/**
 * What detection should look for on this experiment: free-text priorities and the set of
 * focus areas, saved on the experiment so every later run uses them.
 */
export const IssueDetectionSettingsControl = ({ experimentId }: IssueDetectionSettingsControlProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { data: settings, isLoading, error: loadError, save } = useIssueDetectionSettings(experimentId);

  const [isOpen, setIsOpen] = useState(false);
  const [draft, setDraft] = useState<IssueDetectionSettings>(defaultIssueDetectionSettings);
  const [isAddingCategory, setIsAddingCategory] = useState(false);
  const [newCategory, setNewCategory] = useState('');
  const newCategoryRow = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // The field opens below the modal's fold, so without this the caret lands somewhere the
    // user cannot see and they type blind into a box they have to go looking for.
    if (isAddingCategory) {
      // Optional call: jsdom has no `scrollIntoView`, and this is a nicety, not a behaviour.
      newCategoryRow.current?.scrollIntoView?.({ block: 'nearest' });
    }
  }, [isAddingCategory]);

  const customCategories = Object.keys(draft.categories).filter((category) => !isBuiltInCategory(category));
  const trimmedNewCategory = newCategory.trim();
  const isDuplicateCategory = Object.keys(draft.categories).some(
    (category) => category.toLowerCase() === trimmedNewCategory.toLowerCase(),
  );
  const nothingSelected = noFocusAreaSelected(draft);

  const open = useCallback(() => {
    // The draft starts from what is saved, so cancelling leaves the experiment untouched.
    setDraft(settings ?? defaultIssueDetectionSettings());
    setIsAddingCategory(false);
    setNewCategory('');
    save.reset();
    setIsOpen(true);
  }, [save, settings]);

  const close = useCallback(() => {
    if (!save.isLoading) {
      setIsOpen(false);
    }
  }, [save.isLoading]);

  const toggleCategory = useCallback((category: string, selected: boolean) => {
    setDraft((current) => ({ ...current, categories: { ...current.categories, [category]: selected } }));
  }, []);

  const addCategory = useCallback(() => {
    if (!trimmedNewCategory || isDuplicateCategory) {
      return;
    }
    setDraft((current) => ({ ...current, categories: { ...current.categories, [trimmedNewCategory]: true } }));
    setNewCategory('');
    setIsAddingCategory(false);
  }, [isDuplicateCategory, trimmedNewCategory]);

  const removeCategory = useCallback((category: string) => {
    setDraft((current) => {
      const { [category]: _removed, ...categories } = current.categories;
      return { ...current, categories };
    });
  }, []);

  const submit = useCallback(async () => {
    try {
      await save.mutateAsync(draft);
      setIsOpen(false);
    } catch {
      // Reported in the modal, which stays open so the draft can be retried rather than retyped.
    }
  }, [draft, save]);

  const settingsLabel = intl.formatMessage({
    defaultMessage: 'Issue detection settings',
    description: 'Experiment issues tab > Label of the button that opens the detection settings',
  });

  return (
    <>
      <Tooltip componentId={`${COMPONENT_ID}.tooltip`} content={settingsLabel}>
        <Button
          componentId={`${COMPONENT_ID}.open`}
          icon={<GearIcon />}
          aria-label={settingsLabel}
          loading={isLoading}
          disabled={Boolean(loadError)}
          onClick={open}
        />
      </Tooltip>
      <Modal
        componentId={`${COMPONENT_ID}.modal`}
        visible={isOpen}
        title={settingsLabel}
        okText={intl.formatMessage({
          defaultMessage: 'Save',
          description: 'Issue detection settings modal > Label of the button that saves the settings',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Issue detection settings modal > Label of the button that discards the changes',
        })}
        confirmLoading={save.isLoading}
        onOk={() => void submit()}
        onCancel={close}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {save.error && (
            <Alert
              componentId={`${COMPONENT_ID}.save-error`}
              type="error"
              message={intl.formatMessage({
                defaultMessage: 'Could not save these settings. Try again.',
                description: 'Issue detection settings modal > Error shown when saving the settings failed',
              })}
              closable={false}
            />
          )}

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <FormUI.Label htmlFor={`${COMPONENT_ID}.instructions`}>
              {intl.formatMessage({
                defaultMessage: 'Detection priorities',
                description: 'Issue detection settings modal > Label of the free-text priorities field',
              })}
            </FormUI.Label>
            <FormUI.Hint>
              {intl.formatMessage({
                defaultMessage:
                  'Applied every run, when deciding what to flag and how severe it is. Leave empty to use the defaults.',
                description: 'Issue detection settings modal > Hint explaining what the priorities field affects',
              })}
            </FormUI.Hint>
            <Input.TextArea
              id={`${COMPONENT_ID}.instructions`}
              componentId={`${COMPONENT_ID}.instructions`}
              rows={6}
              maxLength={MAX_INSTRUCTIONS_LENGTH}
              value={draft.instructions}
              placeholder={intl.formatMessage(
                {
                  defaultMessage:
                    'For example:{lineBreak}- Treat authentication failures as high severity{lineBreak}- Only report high and medium severity issues{lineBreak}- Ignore cosmetic output formatting issues',
                  description: 'Issue detection settings modal > Placeholder examples for the priorities field',
                },
                { lineBreak: '\n' },
              )}
              onChange={(event) => setDraft((current) => ({ ...current, instructions: event.target.value }))}
            />
          </div>

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <Typography.Title level={4} withoutMargins>
              {intl.formatMessage({
                defaultMessage: 'Focus areas',
                description: 'Issue detection settings modal > Title of the category selection section',
              })}
            </Typography.Title>
            <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
              {ISSUE_CATEGORY_DEFINITIONS.map(({ id, title, description }) => (
                <Tooltip key={id} componentId={`${COMPONENT_ID}.focus-area.tooltip`} content={description}>
                  <ToggleButton
                    componentId={`${COMPONENT_ID}.focus-area.toggle`}
                    pressed={Boolean(draft.categories[id])}
                    onPressedChange={(selected) => toggleCategory(id, selected)}
                  >
                    {title}
                  </ToggleButton>
                </Tooltip>
              ))}
              {customCategories.map((category) => {
                const removeLabel = intl.formatMessage(
                  {
                    defaultMessage: 'Remove custom focus area {category}',
                    description: 'Issue detection settings modal > Label of the button that deletes a custom category',
                  },
                  { category },
                );
                return (
                  // The remove button sits beside the toggle rather than on top of its right
                  // edge: `ToggleButton` sets its own padding, so reserving room inside the
                  // chip with a `css` override loses to the component's own rule and the icon
                  // lands on the last letter of the label.
                  <div key={category} css={{ display: 'inline-flex', alignItems: 'center' }}>
                    <ToggleButton
                      componentId={`${COMPONENT_ID}.custom-focus-area.toggle`}
                      pressed={Boolean(draft.categories[category])}
                      onPressedChange={(selected) => toggleCategory(category, selected)}
                    >
                      {category}
                    </ToggleButton>
                    <Button
                      componentId={`${COMPONENT_ID}.custom-focus-area.remove`}
                      icon={<CloseIcon />}
                      size="small"
                      type="tertiary"
                      aria-label={removeLabel}
                      onClick={() => removeCategory(category)}
                    />
                  </div>
                );
              })}
              {!isAddingCategory && (
                <Button
                  componentId={`${COMPONENT_ID}.custom-focus-area.open`}
                  type="tertiary"
                  icon={<PlusIcon />}
                  onClick={() => setIsAddingCategory(true)}
                >
                  {intl.formatMessage({
                    defaultMessage: 'Custom focus area',
                    description:
                      'Issue detection settings modal > Label of the button that reveals the custom category field',
                  })}
                </Button>
              )}
            </div>

            {nothingSelected && (
              // The run would have nothing to report against, so it uses every area instead.
              // Saying so here keeps that from looking like the settings were ignored.
              <FormUI.Hint>
                {intl.formatMessage({
                  defaultMessage: 'With no focus area selected, detection considers all of them.',
                  description: 'Issue detection settings modal > Hint shown when every focus area has been deselected',
                })}
              </FormUI.Hint>
            )}

            {isAddingCategory && (
              <div ref={newCategoryRow} css={{ display: 'flex', alignItems: 'flex-end', gap: theme.spacing.sm }}>
                <div css={{ display: 'flex', flex: 1, flexDirection: 'column', gap: theme.spacing.xs }}>
                  <FormUI.Label htmlFor={`${COMPONENT_ID}.custom-focus-area.input`}>
                    {intl.formatMessage({
                      defaultMessage: 'Custom focus area',
                      description: 'Issue detection settings modal > Label of the custom category text field',
                    })}
                  </FormUI.Label>
                  <Input
                    id={`${COMPONENT_ID}.custom-focus-area.input`}
                    componentId={`${COMPONENT_ID}.custom-focus-area.input`}
                    autoFocus
                    maxLength={MAX_CATEGORY_LENGTH}
                    value={newCategory}
                    onChange={(event) => setNewCategory(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        event.preventDefault();
                        addCategory();
                      }
                    }}
                  />
                  {trimmedNewCategory && isDuplicateCategory && (
                    <FormUI.Message
                      type="error"
                      message={intl.formatMessage({
                        defaultMessage: 'That focus area already exists.',
                        description:
                          'Issue detection settings modal > Error shown when the typed custom category is already present',
                      })}
                    />
                  )}
                </div>
                <Button
                  componentId={`${COMPONENT_ID}.custom-focus-area.add`}
                  type="primary"
                  disabled={!trimmedNewCategory || isDuplicateCategory}
                  onClick={addCategory}
                >
                  {intl.formatMessage({
                    defaultMessage: 'Add',
                    description: 'Issue detection settings modal > Label of the button that confirms a custom category',
                  })}
                </Button>
                <Button
                  componentId={`${COMPONENT_ID}.custom-focus-area.cancel`}
                  onClick={() => {
                    setNewCategory('');
                    setIsAddingCategory(false);
                  }}
                >
                  {intl.formatMessage({
                    defaultMessage: 'Cancel',
                    description: 'Issue detection settings modal > Label of the button that abandons a custom category',
                  })}
                </Button>
              </div>
            )}
          </div>
        </div>
      </Modal>
    </>
  );
};
