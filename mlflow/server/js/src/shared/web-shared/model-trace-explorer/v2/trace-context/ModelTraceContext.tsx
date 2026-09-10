import { isNil } from 'lodash';
import React, { createContext, useContext, useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTrace, ModelTraceInferenceTableData, ModelTraceSpan } from '../ModelTrace.types';
import { isModelTrace } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerFrameRenderer } from '../frame-renderer/ModelTraceExplorerFrameRenderer';

interface ModelTraceContextType {
  onSelectRow: (row: ModelTraceInferenceTableData | ModelTrace | null) => void;
}

export const ModelTraceContext: React.Context<ModelTraceContextType> = createContext<ModelTraceContextType>({
  onSelectRow: () => {},
});

export const useModelTraceContext = (): ModelTraceContextType => {
  return useContext(ModelTraceContext);
};

const parseTraceTableData = (trace: ModelTraceInferenceTableData | ModelTrace | null): ModelTrace | null => {
  if (isNil(trace)) {
    return null;
  }

  if (isModelTrace(trace)) {
    return trace;
  }

  try {
    const spans = trace.spans?.map((span) => ({
      ...span,
      attributes: JSON.parse(span.attributes),
    })) as ModelTraceSpan[];

    if (isNil(spans)) {
      return null;
    }

    return {
      data: { spans },
      // info is not used in the ModelTraceExplorer component
      info: {},
    };
  } catch (e) {
    return null;
  }
};

/**
 * This context provider is used with the DataGridTable component in
 * the notebook (the component that appears when a user calls display(df)).
 *
 * When users click on a row in the table, a separate hook determines whether
 * the row contains an MLflow trace (see `useSelectDataGridRow.ts`). If it does,
 * the trace is passed to this context provider, which then renders the
 * ModelTraceExplorer component underneath the table.
 */
export const ModelTraceContextProvider = ({ children }: { children: React.ReactNode }): JSX.Element => {
  const [selectedRow, setSelectedRow] = useState<ModelTraceInferenceTableData | ModelTrace | null>(null);
  const modelTrace = parseTraceTableData(selectedRow);
  const { theme } = useDesignSystemTheme();

  return (
    <ModelTraceContext.Provider
      value={{
        onSelectRow: setSelectedRow,
      }}
    >
      {children}
      {!isNil(modelTrace) && (
        <div css={{ borderTop: `1px solid ${theme.colors.border}` }}>
          <ModelTraceExplorerFrameRenderer modelTrace={modelTrace} />
        </div>
      )}
    </ModelTraceContext.Provider>
  );
};
