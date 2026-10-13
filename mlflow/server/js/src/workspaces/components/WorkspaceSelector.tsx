import { useCallback, useMemo, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react';
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
  isGlobalRoute,
  setActiveWorkspace,
  setLastUsedWorkspace,
  useActiveWorkspace,
  validateWorkspaceName,
  WORKSPACE_QUERY_PARAM,
} from '../utils/WorkspaceUtils';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
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
  const queryClient = useQueryClient();
  // Extract workspace from query param. On global routes (e.g. /account)
  // the URL doesn't carry a workspace, so fall back to the in-memory active
  // workspace - that way the trigger keeps showing the user's workspace
  // context across the global-page intermission.
  const workspaceFromUrl = extractWorkspaceFromSearchParams(searchParams);
  const activeWorkspace = useActiveWorkspace();
  const currentWorkspace = workspaceFromUrl ?? activeWorkspace;

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
      // Flip the in-memory singleton that ``FetchUtils`` reads to populate
      // the ``X-MLFLOW-WORKSPACE`` header so requests on the next render
      // already carry the new workspace. ``WorkspaceRouterSync`` will also
      // call this on the URL change below; doing it here is belt-and-
      // suspenders against the brief render between the two.
      setActiveWorkspace(nextWorkspace);

      if (isGlobalRoute(location.pathname)) {
        // Global routes (e.g. /account) are workspace-agnostic - the URL
        // doesn't carry ``?workspace=``, so just stay on the page. The
        // active workspace update above is enough.
        return;
      }

      // Client-side navigation keeps the chrome (sidebar + avatar widget)
      // mounted instead of blanking via ``window.location.reload()``. The
      // sidebar's avatar dropdown was visibly flickering off and back on
      // during a hard reload, since ``/users/current`` had to refetch from
      // scratch before the trigger could render.
      const currentSection = getNavigationSection(location.pathname);
      const targetPath = currentSection || '/';
      navigate(`${targetPath}?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(nextWorkspace)}`);

      // Workspace is a header-level context (not part of any queryKey in
      // this app), so every cached query is now stale relative to the new
      // ``X-MLFLOW-WORKSPACE``. Blanket-invalidate so consumers refetch
      // with the new context.
      queryClient.invalidateQueries();
    },
    [currentWorkspace, location.pathname, navigate, queryClient],
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

  const typedWorkspace = useMemo(() => searchValue.trim(), [searchValue]);
  // Client-side filtering uses the normalized value so leading/trailing spaces
  // don't hide an existing matching workspace.
  const filteredOptions = useMemo(() => {
    if (!typedWorkspace) {
      return options;
    }
    const lowerSearch = typedWorkspace.toLowerCase();
    return options.filter((workspace) => workspace.name.toLowerCase().includes(lowerSearch));
  }, [options, typedWorkspace]);
  const canSubmitTypedWorkspace = useMemo(
    () =>
      typedWorkspace.length > 0 && validateWorkspaceName(typedWorkspace).valid && typedWorkspace !== currentWorkspace,
    [typedWorkspace, currentWorkspace],
  );
  const showTypedWorkspaceOption = useMemo(
    () => canSubmitTypedWorkspace && !options.some((workspace) => workspace.name === typedWorkspace),
    [canSubmitTypedWorkspace, options, typedWorkspace],
  );
  const allowEnterToSubmitTypedWorkspace = showTypedWorkspaceOption && filteredOptions.length === 0;
  const showEmptyState = filteredOptions.length === 0 && typedWorkspace.length > 0 && !showTypedWorkspaceOption;

  const handleTypedWorkspaceSubmit = useCallback(() => {
    if (!canSubmitTypedWorkspace) {
      return;
    }

    handleWorkspaceChange(typedWorkspace);
  }, [canSubmitTypedWorkspace, handleWorkspaceChange, typedWorkspace]);

  const handleSearchKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (event.key !== 'Enter' || !allowEnterToSubmitTypedWorkspace || !(event.target instanceof HTMLInputElement)) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      handleTypedWorkspaceSubmit();
    },
    [allowEnterToSubmitTypedWorkspace, handleTypedWorkspaceSubmit],
  );

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

        <DialogComboboxOptionList>
          <div onKeyDownCapture={handleSearchKeyDown}>
            <DialogComboboxOptionListSearch controlledValue={searchValue} setControlledValue={setSearchValue}>
              {showTypedWorkspaceOption && (
                <DialogComboboxOptionListSelectItem
                  value={typedWorkspace}
                  onChange={() => handleTypedWorkspaceSubmit()}
                  checked={false}
                >
                  Go to workspace "{typedWorkspace}"
                </DialogComboboxOptionListSelectItem>
              )}
              {showEmptyState ? (
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
                        componentId="workspace_selector.tooltip"
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
          </div>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
