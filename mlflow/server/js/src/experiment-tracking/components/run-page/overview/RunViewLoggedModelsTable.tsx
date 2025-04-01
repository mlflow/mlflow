import {
  Alert,
  Button,
  ColumnsIcon,
  getShadowScrollStyles,
  Spacer,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useCombinedRunInputsOutputsModels } from '../../../hooks/logged-models/useCombinedRunInputsOutputsModels';
import { RunInfoEntity } from '../../../types';
import { ExperimentLoggedModelListPageTable } from '../../experiment-logged-models/ExperimentLoggedModelListPageTable';
import {
  ExperimentLoggedModelListPageKnownColumns,
  useExperimentLoggedModelListPageTableColumns,
} from '../../experiment-logged-models/hooks/useExperimentLoggedModelListPageTableColumns';
import { ExperimentLoggedModelOpenDatasetDetailsContextProvider } from '../../experiment-logged-models/hooks/useExperimentLoggedModelOpenDatasetDetails';
import {
  UseGetRunQueryResponseInputs,
  UseGetRunQueryResponseOutputs,
  UseGetRunQueryResponseRunInfo,
} from '../hooks/useGetRunQuery';
import { ExperimentLoggedModelListPageColumnSelector } from '../../experiment-logged-models/ExperimentLoggedModelListPageColumnSelector';
import { first, get } from 'lodash';

const supportedAttributeColumnKeys = [
  ExperimentLoggedModelListPageKnownColumns.RelationshipType,
  ExperimentLoggedModelListPageKnownColumns.Name,
  ExperimentLoggedModelListPageKnownColumns.Status,
  ExperimentLoggedModelListPageKnownColumns.CreationTime,
  ExperimentLoggedModelListPageKnownColumns.Dataset,
];

export const RunViewLoggedModelsTable = ({
  inputs,
  outputs,
  runInfo,
}: {
  inputs?: UseGetRunQueryResponseInputs;
  outputs?: UseGetRunQueryResponseOutputs;
  runInfo?: RunInfoEntity | UseGetRunQueryResponseRunInfo;
}) => {
  const { theme } = useDesignSystemTheme();

  const { models: loggedModels, isLoading, errors } = useCombinedRunInputsOutputsModels(inputs, outputs, runInfo);

  const [columnVisibility, setColumnVisibility] = useState<Record<string, boolean>>({});

  const columnDefs = useExperimentLoggedModelListPageTableColumns({
    loggedModels: loggedModels,
    columnVisibility,
    disablePinnedColumns: true,
    disableOrderBy: true,
    isCompactMode: false,
    supportedAttributeColumnKeys,
  });

  const modelLoadError = useMemo(() => first(errors), [errors]);

  return (
    <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Title level={4} css={{ flexShrink: 0 }}>
          <FormattedMessage
            defaultMessage="Logged models ({length})"
            description="A header for a table of logged models displayed on the run page. The 'length' variable is being replaced with the number of displayed logged models."
            values={{ length: loggedModels.length }}
          />
        </Typography.Title>
        <ExperimentLoggedModelListPageColumnSelector
          columnDefs={columnDefs}
          onUpdateColumns={setColumnVisibility}
          columnVisibility={columnVisibility}
          customTrigger={<Button componentId="mlflow.logged_model.list.columns" icon={<ColumnsIcon />} />}
        />
      </div>
      <Spacer size="sm" shrinks={false} />
      <div
        css={{
          padding: theme.spacing.sm,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.general.borderRadiusBase,
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          overflow: 'hidden',
        }}
      >
        {modelLoadError instanceof Error && modelLoadError.message && (
          <>
            <Alert
              type="error"
              message={modelLoadError.message}
              closable={false}
              componentId="mlflow.run_page.logged_model.list.error"
            />
            <Spacer size="sm" shrinks={false} />
          </>
        )}
        <ExperimentLoggedModelOpenDatasetDetailsContextProvider>
          <ExperimentLoggedModelListPageTable
            columnDefs={columnDefs}
            loggedModels={loggedModels}
            columnVisibility={columnVisibility}
            isLoading={isLoading}
            isLoadingMore={false}
            error={null}
            moreResultsAvailable={false}
            disableLoadMore
            css={getTableTheme(theme)}
            displayShowExampleButton={false}
          />
        </ExperimentLoggedModelOpenDatasetDetailsContextProvider>
      </div>
    </div>
  );
};

const getTableTheme = (theme: Theme) => ({
  '&.ag-theme-balham': {
    '--ag-border-color': theme.colors.border,
    '--ag-row-border-color': theme.colors.border,
    '--ag-foreground-color': theme.colors.textPrimary,
    '--ag-background-color': 'transparent',
    '--ag-odd-row-background-color': 'transparent',
    '--ag-row-hover-color': theme.colors.actionDefaultBackgroundHover,
    '--ag-selected-row-background-color': theme.colors.actionDefaultBackgroundPress,
    '--ag-header-foreground-color': theme.colors.textPrimary,
    '--ag-header-background-color': theme.colors.backgroundPrimary,
    '--ag-modal-overlay-background-color': theme.colors.overlayOverlay,
    '.ag-header-row.ag-header-row-column-group': {
      '--ag-header-foreground-color': theme.colors.textPrimary,
    },
    borderTop: 0,
    fontSize: theme.typography.fontSizeBase,
    '.ag-center-cols-viewport': {
      ...getShadowScrollStyles(theme, {
        orientation: 'horizontal',
      }),
    },
  },
});
