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

  const hasTemplateBraces = /\{\{|\}\}/.test(newCategory.trim());

  useEffect(() => {
    // The field opens below the modal's fold, so without this the caret lands somewhere the
    // user cannot see and they type blind into a box they have to go looking for. Re-run when
    // the validation message appears, because it makes the row taller and would otherwise push
    // itself -- and the Add button -- back out of view.
    if (isAddingCategory) {
      // Optional call: jsdom has no `scrollIntoView`, and this is a nicety, not a behaviour.
      newCategoryRow.current?.scrollIntoView?.({ block: 'nearest' });
    }
  }, [isAddingCategory, hasTemplateBraces]);

  const customCategories = Object.keys(draft.categories).filter((category) => !isBuiltInCategory(category));
  const trimmedNewCategory = newCategory.trim();
  const isDuplicateCategory = Object.keys(draft.categories).some(
    (category) => category.toLowerCase() === trimmedNewCategory.toLowerCase(),
  );
  // A doubled brace in a category name would declare a template variable in the judge's
  // instructions. The server rewrites it rather than failing, which means the name it judges
  // against would no longer be the name stored here -- so the name is rejected at the source
  // instead of being silently altered later.
  const isUnusableCategory = isDuplicateCategory || hasTemplateBraces;
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
    if (!trimmedNewCategory || isUnusableCategory) {
      return;
    }
    setDraft((current) => ({ ...current, categories: { ...current.categories, [trimmedNewCategory]: true } }));
    setNewCategory('');
    setIsAddingCategory(false);
  }, [isUnusableCategory, trimmedNewCategory]);

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
          // No per-experiment write check: OSS has no equivalent of the managed experiment ACL,
          // the same reasoning as `useCanEditExperimentCustomViews`.
          disabled={isLoading || Boolean(loadError)}
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
          {(loadError || save.error) && (
            <Alert
              componentId={`${COMPONENT_ID}.save-error`}
              type="error"
              message={intl.formatMessage({
                defaultMessage: 'Could not save issue detection settings. Try again.',
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
                    defaultMessage: 'Remove custom category {category}',
                    description: 'Issue detection settings modal > Label of the button that deletes a custom category',
                  },
                  { category },
                );
                return (
                  <div key={category} css={{ position: 'relative', display: 'inline-flex' }}>
                    <ToggleButton
                      componentId={`${COMPONENT_ID}.custom-focus-area.toggle`}
                      pressed={Boolean(draft.categories[category])}
                      // Inline rather than `css`: `ToggleButton` sets its own padding, and an
                      // emotion override loses to it, which puts the remove icon on the label.
                      style={{ paddingRight: 40 }}
                      onPressedChange={(selected) => toggleCategory(category, selected)}
                    >
                      {category}
                    </ToggleButton>
                    <Tooltip componentId={`${COMPONENT_ID}.custom-focus-area.remove-tooltip`} content={removeLabel}>
                      <Button
                        componentId={`${COMPONENT_ID}.custom-focus-area.remove`}
                        icon={<CloseIcon />}
                        size="small"
                        type="tertiary"
                        aria-label={removeLabel}
                        style={{ position: 'absolute', top: 4, right: 4 }}
                        onClick={() => removeCategory(category)}
                      />
                    </Tooltip>
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
                    defaultMessage: 'Custom category',
                    description:
                      'Issue detection settings modal > Label of the button that reveals the custom category field',
                  })}
                </Button>
              )}
            </div>

            {nothingSelected && (
              // Not in the managed product. Added because the stored value and the value a run
              // is sent differ in exactly this case, and review asked for that to be visible
              // rather than discovered from the network request.
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
                      defaultMessage: 'Custom category',
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
                  {trimmedNewCategory && hasTemplateBraces && (
                    // Not in the managed product. A duplicate name only disables Add, as there;
                    // this one needs saying, because the name would otherwise be rewritten
                    // server-side and judged under a spelling that is not the one stored.
                    <FormUI.Message
                      type="error"
                      message={intl.formatMessage(
                        {
                          defaultMessage: 'A category name cannot contain {braces}.',
                          description:
                            'Issue detection settings modal > Error shown when the typed custom category contains doubled braces',
                        },
                        { braces: '{{ }}' },
                      )}
                    />
                  )}
                </div>
                <Button
                  componentId={`${COMPONENT_ID}.custom-focus-area.add`}
                  type="primary"
                  disabled={!trimmedNewCategory || isUnusableCategory}
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
