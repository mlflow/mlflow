import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { SELECTED_ISSUE_ID_PARAM } from '../../../constants';

/**
 * Query param-powered hook that returns the selected issue ID and provides a setter.
 * @param onSelect - Optional callback to be invoked when an issue is selected (used for scrolling)
 */
export const useSelectedIssueId = ({ onSelect }: { onSelect?: (issueId: string) => void } = {}) => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedIssueId = searchParams.get(SELECTED_ISSUE_ID_PARAM) ?? undefined;

  const setSelectedIssueId = useCallback(
    (issueId: string | undefined) => {
      setSearchParams(
        (params) => {
          if (issueId === undefined) {
            params.delete(SELECTED_ISSUE_ID_PARAM);
            return params;
          }
          params.set(SELECTED_ISSUE_ID_PARAM, issueId);
          return params;
        },
        { replace: true },
      );

      // Invoke the onSelect callback after updating URL
      if (issueId && onSelect) {
        onSelect(issueId);
      }
    },
    [setSearchParams, onSelect],
  );

  return [selectedIssueId, setSelectedIssueId] as const;
};
