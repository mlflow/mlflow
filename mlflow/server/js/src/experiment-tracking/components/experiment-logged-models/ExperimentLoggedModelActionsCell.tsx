import { Button, DashIcon, DropdownMenu, useDesignSystemTheme, VisibleOffIcon } from '@databricks/design-system';
import { type LoggedModelProto } from '../../types';
import { useExperimentLoggedModelListPageRowVisibilityContext } from './hooks/useExperimentLoggedModelListPageRowVisibility';
import { ReactComponent as VisibleFillIcon } from '../../../common/static/icon-visible-fill.svg';
import { FormattedMessage, useIntl } from 'react-intl';
import { RUNS_VISIBILITY_MODE } from '../experiment-page/models/ExperimentPageUIState';
import { coerceToEnum } from '@databricks/web-shared/utils';

export const ExperimentLoggedModelActionsCell = ({ data, rowIndex }: { data: LoggedModelProto; rowIndex: number }) => {
  const { isRowHidden, toggleRowVisibility } = useExperimentLoggedModelListPageRowVisibilityContext();
  const isHidden = isRowHidden(data.info?.model_id ?? '', rowIndex);
  const { theme } = useDesignSystemTheme();
  return (
    <Button
      componentId="mlflow.logged_model.list_page.row_visibility_toggle"
      type="link"
      onClick={() => toggleRowVisibility(data.info?.model_id ?? '', rowIndex)}
      icon={
        isHidden ? (
          <VisibleOffIcon css={{ color: theme.colors.textSecondary }} />
        ) : (
          <VisibleFillIcon css={{ color: theme.colors.textSecondary }} />
        )
      }
    />
  );
};

export const ExperimentLoggedModelActionsHeaderCell = () => {
  const intl = useIntl();
  const { visibilityMode, usingCustomVisibility, setRowVisibilityMode } =
    useExperimentLoggedModelListPageRowVisibilityContext();
  const { theme } = useDesignSystemTheme();
  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId="mlflow.logged_model.list_page.global_row_visibility_toggle"
          type="link"
          data-testid="experiment-view-runs-visibility-column-header"
          aria-label={intl.formatMessage({
            defaultMessage: 'Toggle visibility of rows',
            description:
              'Accessibility label for the button that toggles visibility of rows in the experiment view logged models compare mode',
          })}
        >
          {visibilityMode === RUNS_VISIBILITY_MODE.HIDEALL ? (
            <VisibleOffIcon css={{ color: theme.colors.textSecondary }} />
          ) : (
            <VisibleFillIcon css={{ color: theme.colors.textSecondary }} />
          )}
        </Button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Content>
        <DropdownMenu.RadioGroup
          componentId="mlflow.logged_model.list_page.global_row_visibility_toggle.options"
          value={visibilityMode}
          onValueChange={(e) =>
            setRowVisibilityMode(coerceToEnum(RUNS_VISIBILITY_MODE, e, RUNS_VISIBILITY_MODE.FIRST_10_RUNS))
          }
        >
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.FIRST_10_RUNS}>
            {/* Dropdown menu does not support indeterminate state, so we're doing it manually */}
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Show first 10"
              description="Menu option for showing only 10 first runs in the experiment view runs compare mode"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.FIRST_20_RUNS}>
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Show first 20"
              description="Menu option for showing only 10 first runs in the experiment view runs compare mode"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.SHOWALL}>
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Show all runs"
              description="Menu option for revealing all hidden runs in the experiment view runs compare mode"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.HIDEALL}>
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Hide all runs"
              description="Menu option for revealing all hidden runs in the experiment view runs compare mode"
            />
          </DropdownMenu.RadioItem>
        </DropdownMenu.RadioGroup>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
