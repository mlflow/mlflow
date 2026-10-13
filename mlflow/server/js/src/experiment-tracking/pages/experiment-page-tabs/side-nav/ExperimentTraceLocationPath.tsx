import { Button, CatalogIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG } from '../../../constants';
import { COLLAPSED_CLASS_NAME, FULL_WIDTH_CLASS_NAME } from './constants';

/**
 * Displays the trace destination path in the sidebar when the experiment
 * uses a UC trace location (2-part UC schema or 3-part table prefix path).
 * Hidden for V3 experiments (no destination path tag).
 */
export const ExperimentTraceLocationPath = () => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams();

  const { data: experimentData, loading } = useGetExperimentQuery({
    experimentId: experimentId ?? '',
  });

  if (loading || !experimentData) {
    return null;
  }

  const tags = experimentData && 'tags' in experimentData ? experimentData?.tags : [];
  const destinationPath = tags?.find((tag) => tag.key === MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG)?.value;

  if (!destinationPath) {
    return null;
  }

  const parts = destinationPath.split('.');
  const catalogName = parts[0];
  const schemaName = parts[1];
  const schemaPath =
    catalogName && schemaName
      ? `/explore/data/${encodeURIComponent(catalogName)}/${encodeURIComponent(schemaName)}`
      : undefined;

  const catalogIcon = (
    <CatalogIcon
      css={{
        flexShrink: 0,
        color: theme.colors.textPrimary,
        '& svg': { width: 12, height: 12 },
        width: 12,
        height: 12,
      }}
    />
  );

  return (
    <>
      {/* Collapsed view: icon only with tooltip */}
      <div
        className={COLLAPSED_CLASS_NAME}
        css={{
          justifyContent: 'center',
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        }}
      >
        <Tooltip content={destinationPath} side="right" componentId="mlflow.experiment.trace_location_path.tooltip">
          <span css={{ display: 'flex' }}>{catalogIcon}</span>
        </Tooltip>
      </div>
      {/* Expanded view: icon + text with tooltip */}
      <Tooltip content={destinationPath} side="right" componentId="mlflow.experiment.trace_location_path.tooltip">
        <div
          className={FULL_WIDTH_CLASS_NAME}
          css={{
            padding: `${theme.spacing.xs}px ${theme.spacing.sm + 6}px`,
            paddingBottom: `${theme.spacing.sm}px`,
            margin: `0 -${theme.spacing.sm}px`,
            borderBottom: `1px solid ${theme.colors.border}`,
            overflow: 'hidden',
            minWidth: 0,
          }}
        >
          <Button
            componentId="mlflow.experiment.trace_location_path.button"
            type="link"
            icon={catalogIcon}
            onClick={() => {
              if (schemaPath) {
                window.open(schemaPath, '_blank', 'noopener,noreferrer');
              }
            }}
            css={{
              '&&': {
                all: 'unset !important' as any,
                cursor: `${schemaPath ? 'pointer' : 'default'} !important` as any,
                display: 'flex !important' as any,
                alignItems: 'center !important' as any,
                gap: '0 !important' as any,
                overflow: 'hidden !important' as any,
                textOverflow: 'ellipsis !important' as any,
                whiteSpace: 'nowrap !important' as any,
                minWidth: '0 !important' as any,
                fontSize: `${theme.typography.fontSizeSm}px !important` as any,
                color: `${theme.colors.textPrimary} !important` as any,
              },
              '&& > span:not(.anticon)': {
                display: 'inline !important' as any,
                overflow: 'hidden !important' as any,
                textOverflow: 'ellipsis !important' as any,
                color: 'inherit !important' as any,
              },
              '&&:hover': schemaPath
                ? {
                    textDecoration: 'underline !important' as any,
                    color: `${theme.colors.textPrimary} !important` as any,
                  }
                : undefined,
            }}
          >
            {destinationPath}
          </Button>
        </div>
      </Tooltip>
    </>
  );
};
