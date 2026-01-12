import {
  CUSTOM_METADATA_COLUMN_ID,
  FilterOperator,
  HiddenFilterOperator,
} from '@databricks/web-shared/genai-traces-table';
import { SESSION_ID_METADATA_KEY } from '@databricks/web-shared/model-trace-explorer';

export const getChatSessionsFilter = ({
  sessionId,
}: {
  // pass null to query all sessions
  sessionId: string | null;
}) => {
  const queryAllOperator = FilterOperator.CONTAINS;

  return [
    {
      column: `${CUSTOM_METADATA_COLUMN_ID}:${SESSION_ID_METADATA_KEY}`,
      operator: sessionId ? FilterOperator.EQUALS : queryAllOperator,
      value: sessionId ?? '',
    },
  ];
};
