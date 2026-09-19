import {
  ApplyDesignSystemContextOverrides,
  Button,
  ChevronDownIcon,
  DropdownMenu,
  GridIcon,
  PlusIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type { ModelTraceExplorerDisplayMode } from './ModelTraceExplorerContext';
import { useOptionalCustomViewDefinition } from '../custom-view/CustomViewDefinitionContext';

const DEFAULT_VIEW_VALUE = 'default';
const CUSTOM_VIEW_VALUE_PREFIX = 'custom:';

export const ModelTraceExplorerCustomViewSelector = ({
  value,
  onValueChange,
  onCreateCustomView,
  isCustomViewEnabled,
  canCreateCustomView,
  size,
  compact = false,
  // The selector is mounted on multiple surfaces (the review modal and the trace
  // drawer). Each mount passes its own componentId so their interaction telemetry
  // stays distinguishable. Defaults to the review-modal id for backwards compatibility.
  componentId = 'mlflow.evaluations_review.modal.custom_view_selector',
}: {
  value: ModelTraceExplorerDisplayMode;
  onValueChange: (value: ModelTraceExplorerDisplayMode) => void;
  onCreateCustomView: () => void;
  isCustomViewEnabled: boolean;
  canCreateCustomView: boolean;
  size?: 'small';
  compact?: boolean;
  componentId?: string;
}): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const customViewDefinition = useOptionalCustomViewDefinition();
  const selectedValue = isCustomViewEnabled ? value : 'default';
  const defaultViewLabel = intl.formatMessage({
    defaultMessage: 'Default view',
    description: 'Selected trace explorer default view mode label',
  });
  const customViewLabel = intl.formatMessage({
    defaultMessage: 'Custom view',
    description: 'Selected trace explorer custom view mode label',
  });
  const untitledCustomViewLabel = intl.formatMessage({
    defaultMessage: 'Untitled custom view',
    description: 'Fallback label for an unnamed custom trace view in the trace explorer view selector',
  });
  const selectedValueLabel =
    selectedValue === 'custom' ? customViewDefinition?.activeView?.name || customViewLabel : defaultViewLabel;
  const selectViewLabel = intl.formatMessage({
    defaultMessage: 'Select view',
    description: 'Tooltip for the trace explorer view selector',
  });
  const selectedMenuValue =
    selectedValue === 'default'
      ? DEFAULT_VIEW_VALUE
      : customViewDefinition?.activeViewId
        ? `${CUSTOM_VIEW_VALUE_PREFIX}${customViewDefinition.activeViewId}`
        : undefined;
  const handleValueChange = (nextValue: string) => {
    if (nextValue === DEFAULT_VIEW_VALUE) {
      onValueChange('default');
      return;
    }
    if (nextValue.startsWith(CUSTOM_VIEW_VALUE_PREFIX)) {
      customViewDefinition?.selectView(nextValue.slice(CUSTOM_VIEW_VALUE_PREFIX.length));
      onValueChange('custom');
    }
  };
  const subtleButtonBorderCss = {
    '&&, &&:hover, &&:active': {
      borderColor: `${theme.colors.borderDecorative} !important`,
    },
  };

  return (
    <ApplyDesignSystemContextOverrides getPopupContainer={() => document.body}>
      <DropdownMenu.Root>
        <Tooltip componentId={`${componentId}.tooltip`} content={selectViewLabel}>
          <DropdownMenu.Trigger asChild>
            <Button
              componentId={componentId}
              icon={
                <GridIcon
                  css={{
                    '&&': { color: `${theme.colors.textSecondary} !important` },
                    svg: { width: 15, height: 15 },
                  }}
                />
              }
              endIcon={
                compact ? undefined : (
                  <ChevronDownIcon css={{ color: theme.colors.textSecondary, svg: { width: 12, height: 12 } }} />
                )
              }
              size={size}
              aria-label={compact ? selectedValueLabel : undefined}
              css={[{ flexShrink: 0 }, subtleButtonBorderCss]}
            >
              {compact ? undefined : selectedValueLabel}
            </Button>
          </DropdownMenu.Trigger>
        </Tooltip>
        <DropdownMenu.Content align="start" onCloseAutoFocus={(event) => event.preventDefault()}>
          <DropdownMenu.RadioGroup
            componentId={`${componentId}.radio`}
            value={selectedMenuValue}
            onValueChange={handleValueChange}
          >
            <DropdownMenu.RadioItem value={DEFAULT_VIEW_VALUE}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage defaultMessage="Default view" description="Trace explorer default view menu item" />
            </DropdownMenu.RadioItem>
            {isCustomViewEnabled &&
              customViewDefinition?.views.map((view) => (
                <DropdownMenu.RadioItem key={view.id} value={`${CUSTOM_VIEW_VALUE_PREFIX}${view.id}`}>
                  <DropdownMenu.ItemIndicator />
                  {view.name || untitledCustomViewLabel}
                </DropdownMenu.RadioItem>
              ))}
          </DropdownMenu.RadioGroup>
          {canCreateCustomView && (
            <>
              <DropdownMenu.Separator />
              <DropdownMenu.Item componentId={`${componentId}.create`} onClick={onCreateCustomView}>
                <PlusIcon css={{ marginRight: theme.spacing.sm }} />
                <FormattedMessage
                  defaultMessage="Create custom view"
                  description="Menu item that opens the custom trace view authoring surface"
                />
              </DropdownMenu.Item>
            </>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </ApplyDesignSystemContextOverrides>
  );
};
