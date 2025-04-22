import { Button, ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';
import { isString } from 'lodash';
import { useCallback, useMemo, useState } from 'react';
import { MlflowService } from '../../sdk/MlflowService';
import Utils from '../../../common/utils/Utils';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import type { CellContext, ColumnDefTemplate } from '@tanstack/react-table';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { getTraceInfoInputs, getTraceInfoOutputs, isTraceMetadataPossiblyTruncated } from './TracesView.utils';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { css } from '@emotion/react';

const clampedLinesCss = css`
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
`;

const TracesViewTablePreviewCell = ({
  value,
  traceId,
  previewFieldName,
}: {
  value: string;
  traceId: string;
  previewFieldName: 'request' | 'response';
}) => {
  const { theme } = useDesignSystemTheme();
  const [isExpanded, setIsExpanded] = useState(false);
  const [fullData, setFullData] = useState<string | null>(null);

  const [loading, setLoading] = useState<boolean>(false);

  const fetchFullData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await MlflowService.getExperimentTraceData<{
        request?: any;
        response?: any;
      }>(traceId);

      if (previewFieldName in response) {
        const previewValue = response[previewFieldName];
        const requestData = isString(previewValue) ? previewValue : JSON.stringify(previewValue);
        setFullData(requestData);
      }
    } catch (e: any) {
      const errorMessage = e instanceof ErrorWrapper ? e.getUserVisibleError() : e.message;
      Utils.logErrorAndNotifyUser(`Error fetching response: ${errorMessage}`);
    }
    setLoading(false);
  }, [previewFieldName, traceId]);

  const valuePossiblyTruncated = isTraceMetadataPossiblyTruncated(value);

  const expand = useCallback(async () => {
    if (!fullData && valuePossiblyTruncated) {
      await fetchFullData();
    }
    setIsExpanded(true);
  }, [fullData, fetchFullData, valuePossiblyTruncated]);

  const collapse = useCallback(() => {
    setIsExpanded(false);
  }, []);

  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs }}>
      <Button
        // it's difficult to distinguish between run and experiment page
        // in this component due to how the data is passed to the table,
        // so the base component ID here is simply `mlflow.traces`
        componentId="mlflow.traces.traces_table.expand_cell_preview"
        size="small"
        icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={isExpanded ? collapse : expand}
        css={{ flexShrink: 0 }}
        loading={loading}
        type="primary"
      />
      <div
        title={value}
        css={[
          {
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          },
          !isExpanded && clampedLinesCss,
        ]}
      >
        {isExpanded ? <ExpandedParamCell value={fullData ?? value} /> : value}
      </div>
    </div>
  );
};

const ExpandedParamCell = ({ value }: { value: string }) => {
  const { theme } = useDesignSystemTheme();

  const structuredJSONValue = useMemo(() => {
    // Attempts to parse the value as JSON and returns a pretty printed version if successful.
    // If JSON structure is not found, returns null.
    try {
      const objectData = JSON.parse(value);
      return JSON.stringify(objectData, null, 2);
    } catch (e) {
      return null;
    }
  }, [value]);
  return (
    <div
      css={{
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontFamily: structuredJSONValue ? 'monospace' : undefined,
      }}
    >
      <CodeSnippet
        language="json"
        wrapLongLines
        style={{
          padding: theme.spacing.sm,
        }}
        theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
      >
        {structuredJSONValue || value}
      </CodeSnippet>
    </div>
  );
};

export const TracesViewTableRequestPreviewCell: ColumnDefTemplate<CellContext<ModelTraceInfoWithRunName, unknown>> = ({
  row: { original },
}) => (
  <TracesViewTablePreviewCell
    previewFieldName="request"
    traceId={original.request_id || ''}
    value={getTraceInfoInputs(original) || ''}
  />
);

export const TracesViewTableResponsePreviewCell: ColumnDefTemplate<CellContext<ModelTraceInfoWithRunName, unknown>> = ({
  row: { original },
}) => (
  <TracesViewTablePreviewCell
    previewFieldName="response"
    traceId={original.request_id || ''}
    value={getTraceInfoOutputs(original) || ''}
  />
);
