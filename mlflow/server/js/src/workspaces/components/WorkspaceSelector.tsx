import { useCallback, useMemo, useState } from 'react';
import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxTrigger,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  useDesignSystemTheme,
  Tooltip,
} from '@databricks/design-system';
import { uniqBy } from 'lodash';

import { shouldEnableWorkspaces } from '../../common/utils/FeatureUtils';
import {
  extractWorkspaceFromSearchParams,
  setLastUsedWorkspace,
  WORKSPACE_QUERY_PARAM,
} from '../../workspaces/utils/WorkspaceUtils';
import { useLocation, useNavigate, useSearchParams } from '../../common/utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';
import { useWorkspaces } from '../hooks/useWorkspaces';

export const WorkspaceSelector = () => {
  const workspacesEnabled = shouldEnableWorkspaces();
  const [searchValue, setSearchValue] = useState('');
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate({ bypassWorkspacePrefix: true });
  // Extract workspace from query param
  const currentWorkspace = extractWorkspaceFromSearchParams(searchParams);

  // Fetch workspaces using custom hook
  const { workspaces, isLoading, isError, refetch } = useWorkspaces(workspacesEnabled);

  // Smart redirect - preserve navigation context
  const getNavigationSection = (pathname: string): string => {
    if (pathname.includes('/experiments')) return '/experiments';
    if (pathname.includes('/models')) return '/models';
    if (pathname.includes('/prompts')) return '/prompts';
    return '';
  };

  const handleWorkspaceChange = useCallback(
    (nextWorkspace?: string) => {
      if (!nextWorkspace || nextWorkspace === currentWorkspace) {
        return;
      }

      // Persist to localStorage for UI hints
      setLastUsedWorkspace(nextWorkspace);

      // Hard reload to cleanly switch workspace context (clears all caches)
      const currentSection = getNavigationSection(location.pathname);
      const targetPath = currentSection || '/';
      window.location.hash = `#${targetPath}?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(nextWorkspace)}`;
      window.location.reload();
    },
    [currentWorkspace, location.pathname],
  );

  // Refresh workspaces when combobox is opened to catch label selector changes
  const handleOpenChange = (isOpen: boolean) => {
    if (isOpen) {
      refetch();
    }
  };

  const options = useMemo(() => {
    const allWorkspaces = [...workspaces];

    // Add current workspace if it's not in the list (might be invalid)
    if (currentWorkspace && !workspaces.some((w) => w.name === currentWorkspace)) {
      allWorkspaces.push({ name: currentWorkspace, description: null });
    }

    return uniqBy(allWorkspaces, 'name');
  }, [workspaces, currentWorkspace]);

  // Client-side filtering
  const filteredOptions = useMemo(() => {
    if (!searchValue) {
      return options;
    }
    const lowerSearch = searchValue.toLowerCase();
    return options.filter((workspace) => workspace.name.toLowerCase().includes(lowerSearch));
  }, [options, searchValue]);

  if (!workspacesEnabled) {
    return null;
  }

  return (
    <DialogCombobox
      componentId="workspace_selector"
      label="Workspace"
      value={!currentWorkspace ? [] : [currentWorkspace]}
      onOpenChange={handleOpenChange}
    >
      <DialogComboboxTrigger
        withInlineLabel={false}
        placeholder="Select workspace"
        renderDisplayedValue={() => currentWorkspace}
        onClear={() => navigate(Routes.rootRoute)}
        width="100%"
      />
      <DialogComboboxContent style={{ zIndex: theme.options.zIndexBase + 100 }} loading={isLoading}>
        {isError && (
          <div css={{ padding: theme.spacing.sm, color: theme.colors.textValidationDanger }}>
            Failed to load workspaces
          </div>
        )}

        {!isError && (
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSearch onSearch={(value) => setSearchValue(value)}>
              {filteredOptions.length === 0 && searchValue ? (
                // Provide a dummy item when no results to prevent crash
                <DialogComboboxOptionListSelectItem value="" onChange={() => {}} checked={false} disabled>
                  No workspaces found
                </DialogComboboxOptionListSelectItem>
              ) : (
                filteredOptions.map((workspace) => {
                  const item = (
                    <DialogComboboxOptionListSelectItem
                      key={workspace.name}
                      value={workspace.name}
                      onChange={(value) => handleWorkspaceChange(value)}
                      checked={workspace.name === currentWorkspace}
                    >
                      {workspace.name}
                    </DialogComboboxOptionListSelectItem>
                  );

                  // Wrap with Tooltip if workspace has description
                  if (workspace.description) {
                    return (
                      <Tooltip
                        key={workspace.name}
                        componentId={`workspace_selector.tooltip.${workspace.name}`}
                        content={workspace.description}
                        side="right"
                      >
                        {item}
                      </Tooltip>
                    );
                  }

                  return item;
                })
              )}
            </DialogComboboxOptionListSearch>
          </DialogComboboxOptionList>
        )}
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
