import {
  Button,
  CloseIcon,
  FormUI,
  Input,
  PlusIcon,
  ToggleButton,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import { useIntl } from 'react-intl';
import { noFocusAreaSelected, type IssueDetectionSettings } from './issueDetectionSettings';
import {
  ALL_ISSUE_CATEGORIES,
  ISSUE_CATEGORY_DEFINITIONS,
} from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';

const COMPONENT_ID = 'mlflow.experiment-issues.detection-settings';
const isBuiltInCategory = (category: string) => ALL_ISSUE_CATEGORIES.some((id) => id === category);

export const IssueDetectionSettingsForm = ({
  draft,
  setDraft,
  disabled = false,
}: {
  draft: IssueDetectionSettings;
  setDraft: Dispatch<SetStateAction<IssueDetectionSettings>>;
  disabled?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isAddingCustomCategory, setIsAddingCustomCategory] = useState(false);
  const [customCategory, setCustomCategory] = useState('');
  const customCategoryRow = useRef<HTMLDivElement>(null);
  const customCategories = Object.keys(draft.categories).filter((category) => !isBuiltInCategory(category));
  const normalizedCustomCategory = customCategory.trim();
  const isCustomCategoryTaken =
    ALL_ISSUE_CATEGORIES.some((category) => category === normalizedCustomCategory.toLowerCase()) ||
    Object.keys(draft.categories).some((category) => category.toLowerCase() === normalizedCustomCategory.toLowerCase());
  // Added for OSS. A doubled brace here declares a template variable in the judge's
  // instructions. The server neutralizes it rather than failing, which means the name it judges
  // against would no longer be the name stored here -- so it is refused at the source.
  const hasTemplateBraces = /\{\{|\}\}/.test(normalizedCustomCategory);
  const isCustomCategoryUnusable = isCustomCategoryTaken || hasTemplateBraces;

  useEffect(() => {
    // Added for OSS. The field opens below the modal's fold, so without this the caret lands
    // somewhere the user cannot see. Re-runs when the validation message appears, because it
    // makes the row taller and would otherwise push itself -- and Add -- back out of view.
    if (isAddingCustomCategory) {
      // Optional call: jsdom has no `scrollIntoView`, and this is a nicety, not a behaviour.
      customCategoryRow.current?.scrollIntoView?.({ block: 'nearest' });
    }
  }, [isAddingCustomCategory, hasTemplateBraces]);

  const toggleCategory = useCallback(
    (category: string, selected: boolean) => {
      setDraft((current) => ({
        ...current,
        categories: { ...current.categories, [category]: selected },
      }));
    },
    [setDraft],
  );

  const addCustomCategory = useCallback(() => {
    if (!normalizedCustomCategory || isCustomCategoryUnusable) {
      return;
    }
    setDraft((current) => ({
      ...current,
      categories: { ...current.categories, [normalizedCustomCategory]: true },
    }));
    setCustomCategory('');
    setIsAddingCustomCategory(false);
  }, [isCustomCategoryUnusable, normalizedCustomCategory, setDraft]);

  const removeCustomCategory = useCallback(
    (category: string) => {
      setDraft((current) => {
        const categories = { ...current.categories };
        delete categories[category];
        return { ...current, categories };
      });
    },
    [setDraft],
  );

  return (
    <fieldset
      disabled={disabled}
      css={{
        border: 0,
        margin: 0,
        padding: 0,
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.lg,
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <FormUI.Label htmlFor={`${COMPONENT_ID}.instructions`}>
          {intl.formatMessage({
            defaultMessage: 'Detection priorities',
            description: 'Issue detection settings modal > Detection priorities field label',
          })}
        </FormUI.Label>
        <Input.TextArea
          id={`${COMPONENT_ID}.instructions`}
          componentId={`${COMPONENT_ID}.instructions`}
          rows={6}
          maxLength={5000}
          value={draft.instructions}
          placeholder={intl.formatMessage(
            {
              defaultMessage:
                'For example:{lineBreak}- Treat authentication failures as high severity{lineBreak}- Only report high and medium severity issues{lineBreak}- Ignore cosmetic output formatting issues',
              description: 'Issue detection settings modal > Detection instructions placeholder',
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
            description: 'Issue detection settings modal > Focus areas section title',
          })}
        </Typography.Title>
        <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
          {ISSUE_CATEGORY_DEFINITIONS.map(({ id, title, description }) => (
            <Tooltip key={id} componentId="mlflow.issue-detection-settings.focus-area.tooltip" content={description}>
              <ToggleButton
                componentId="mlflow.issue-detection-settings.focus-area.toggle"
                pressed={Boolean(draft.categories[id])}
                onPressedChange={(selected) => toggleCategory(id, selected)}
              >
                {title}
              </ToggleButton>
            </Tooltip>
          ))}
          {customCategories.map((category) => {
            const removeCategoryLabel = intl.formatMessage(
              {
                defaultMessage: 'Remove custom category {category}',
                description: 'Issue detection settings modal > Remove custom category button label',
              },
              { category },
            );
            return (
              <div key={category} css={{ position: 'relative', display: 'inline-flex' }}>
                <ToggleButton
                  componentId="mlflow.issue-detection-settings.custom-category.toggle"
                  pressed={Boolean(draft.categories[category])}
                  style={{ paddingRight: 40 }}
                  onPressedChange={(selected) => toggleCategory(category, selected)}
                >
                  {category}
                </ToggleButton>
                <Tooltip
                  componentId="mlflow.issue-detection-settings.custom-category.remove-tooltip"
                  content={removeCategoryLabel}
                >
                  <Button
                    componentId="mlflow.issue-detection-settings.custom-category.remove"
                    icon={<CloseIcon />}
                    size="small"
                    type="tertiary"
                    aria-label={removeCategoryLabel}
                    style={{ position: 'absolute', top: 4, right: 4 }}
                    onClick={() => removeCustomCategory(category)}
                  />
                </Tooltip>
              </div>
            );
          })}
          {noFocusAreaSelected(draft) && (
            // Added for OSS. Deselecting everything is the one case where the stored settings
            // and what a run is sent differ, and review asked for that to be visible rather
            // than discovered from the network request.
            <FormUI.Hint>
              {intl.formatMessage({
                defaultMessage: 'Select at least one focus area to save these settings.',
                description: 'Issue detection settings modal > Hint shown when every focus area has been deselected',
              })}
            </FormUI.Hint>
          )}
          {!isAddingCustomCategory && (
            <Button
              componentId={`${COMPONENT_ID}.custom_category.open`}
              type="tertiary"
              icon={<PlusIcon />}
              onClick={() => setIsAddingCustomCategory(true)}
            >
              {intl.formatMessage({
                defaultMessage: 'Custom category',
                description: 'Issue detection settings modal > Add custom category button',
              })}
            </Button>
          )}
        </div>
        {isAddingCustomCategory && (
          <div ref={customCategoryRow} css={{ display: 'flex', alignItems: 'flex-end', gap: theme.spacing.sm }}>
            <div css={{ display: 'flex', flex: 1, flexDirection: 'column', gap: theme.spacing.xs }}>
              <FormUI.Label htmlFor={`${COMPONENT_ID}.custom_category.input`}>
                {intl.formatMessage({
                  defaultMessage: 'Custom category',
                  description: 'Issue detection settings modal > Custom category field label',
                })}
              </FormUI.Label>
              <Input
                id={`${COMPONENT_ID}.custom_category.input`}
                componentId={`${COMPONENT_ID}.custom_category.input`}
                autoFocus
                maxLength={80}
                value={customCategory}
                onChange={(event) => setCustomCategory(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    addCustomCategory();
                  }
                }}
              />
              {normalizedCustomCategory && hasTemplateBraces && (
                // Added for OSS. A taken name only disables Add, as upstream; this one
                // needs saying, because the name would otherwise be neutralized server-side
                // and judged under a spelling that is not the one stored.
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
              componentId={`${COMPONENT_ID}.custom_category.add`}
              type="primary"
              disabled={!normalizedCustomCategory || isCustomCategoryUnusable}
              onClick={addCustomCategory}
            >
              {intl.formatMessage({
                defaultMessage: 'Add',
                description: 'Issue detection settings modal > Confirm custom category button',
              })}
            </Button>
            <Button
              componentId={`${COMPONENT_ID}.custom_category.cancel`}
              onClick={() => {
                setCustomCategory('');
                setIsAddingCustomCategory(false);
              }}
            >
              {intl.formatMessage({
                defaultMessage: 'Cancel',
                description: 'Issue detection settings modal > Cancel custom category button',
              })}
            </Button>
          </div>
        )}
      </div>
    </fieldset>
  );
};
