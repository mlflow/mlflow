import React, { useEffect, useMemo, useState } from 'react';
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

import { shouldEnableWorkspaces } from '../utils/FeatureUtils';
import { fetchAPI, getAjaxUrl } from '../utils/FetchUtils';
import { DEFAULT_WORKSPACE_NAME, extractWorkspaceFromPathname, setActiveWorkspace, setAvailableWorkspaces } from '../utils/WorkspaceUtils';
import { useLocation, useNavigate } from '../utils/RoutingUtils';

type Workspace = {
  name: string;
  description?: string | null;
};

const WORKSPACES_ENDPOINT = 'ajax-api/2.0/mlflow/workspaces';

export const WorkspaceSelector = () => {
  const workspacesEnabled = shouldEnableWorkspaces();
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);
  const [searchValue, setSearchValue] = useState('');
  const location = useLocation();
  const navigate = useNavigate();
  const { theme } = useDesignSystemTheme();

  const workspaceFromPath = extractWorkspaceFromPathname(location.pathname);
  const currentWorkspace = workspaceFromPath ?? DEFAULT_WORKSPACE_NAME;

  // Handle case where current workspace is no longer available (e.g., label selector changed)
  useEffect(() => {
    if (workspaces.length > 0 && currentWorkspace && !workspaces.find(w => w.name === currentWorkspace)) {
      // Current workspace is no longer in the list - redirect to first available workspace
      const fallback = workspaces.find(w => w.name === DEFAULT_WORKSPACE_NAME) ?? workspaces[0];
      if (fallback) {
        handleWorkspaceChange(fallback.name);
      }
    }
  }, [workspaces, currentWorkspace]);

  const loadWorkspaces = async () => {
    if (!workspacesEnabled) {
      setWorkspaces([]);
      setLoadFailed(false);
      setAvailableWorkspaces([]);
      return;
    }

    setLoading(true);
    setLoadFailed(false);
    try {
      const response = await fetchAPI(getAjaxUrl(WORKSPACES_ENDPOINT));

      const fetched = Array.isArray(response?.workspaces) ? response.workspaces : [];
      const filteredWorkspaces: Workspace[] = [];
      for (const item of fetched as Array<Workspace | Record<string, unknown>>) {
        if (item && typeof (item as Workspace)?.name === 'string') {
          const workspaceItem = item as Workspace;
          filteredWorkspaces.push({
            name: workspaceItem.name,
            description: workspaceItem.description ?? null,
          });
        }
      }
      setWorkspaces(filteredWorkspaces);
      // Store available workspaces for access validation
      setAvailableWorkspaces(filteredWorkspaces.map(w => w.name));
    } catch {
      setLoadFailed(true);
      setAvailableWorkspaces([]);
    } finally {
      setLoading(false);
    }
  };

  // Load workspaces on mount and when workspace selector is opened
  useEffect(() => {
    loadWorkspaces().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspacesEnabled]);

  // Refresh workspaces when combobox is opened to catch label selector changes
  const handleOpenChange = (isOpen: boolean) => {
    if (isOpen) {
      loadWorkspaces().catch(() => undefined);
    }
  };

  const options = useMemo(() => {
    const deduped = new Map<string, Workspace>();

    for (const workspace of workspaces) {
      deduped.set(workspace.name, workspace);
    }

    if (deduped.size === 0) {
      deduped.set(DEFAULT_WORKSPACE_NAME, { name: DEFAULT_WORKSPACE_NAME, description: null });
    }

    if (currentWorkspace && !deduped.has(currentWorkspace)) {
      deduped.set(currentWorkspace, { name: currentWorkspace, description: null });
    }

    return Array.from(deduped.values());
  }, [workspaces, currentWorkspace]);

  // Client-side filtering
  const filteredOptions = useMemo(() => {
    if (!searchValue) {
      return options;
    }
    const lowerSearch = searchValue.toLowerCase();
    return options.filter((workspace) => workspace.name.toLowerCase().includes(lowerSearch));
  }, [options, searchValue]);

  // Smart redirect - preserve navigation context
  const getNavigationSection = (pathname: string): string => {
    if (pathname.includes('/experiments')) return '/experiments';
    if (pathname.includes('/models')) return '/models';
    if (pathname.includes('/prompts')) return '/prompts';
    return '';
  };

  const handleWorkspaceChange = (nextWorkspace?: string) => {
    if (!nextWorkspace || nextWorkspace === currentWorkspace) {
      return;
    }

    const encodedWorkspace = encodeURIComponent(nextWorkspace);
    setActiveWorkspace(nextWorkspace);
    
    // Smart redirect - preserve navigation section
    const currentSection = getNavigationSection(location.pathname);
    const targetPath = `/workspaces/${encodedWorkspace}${currentSection}`;
    
    navigate(targetPath);
    setSearchValue(''); // Clear search on selection
  };

  if (!workspacesEnabled) {
    return null;
  }

  return (
    <DialogCombobox
      componentId="workspace_selector"
      label="Workspace"
      value={[currentWorkspace]}
      onOpenChange={handleOpenChange}
    >
      <DialogComboboxTrigger
        withInlineLabel={false}
        placeholder="Select workspace"
        renderDisplayedValue={() => currentWorkspace}
        allowClear={false}
      />
      <DialogComboboxContent
        style={{ zIndex: theme.options.zIndexBase + 100 }}
      loading={loading}
      >
        {loadFailed && (
          <div css={{ padding: theme.spacing.sm, color: theme.colors.textValidationDanger }}>
            Failed to load workspaces
          </div>
        )}
        
        {!loadFailed && (
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSearch onSearch={(value) => setSearchValue(value)}>
              {filteredOptions.length === 0 && searchValue ? (
                // Provide a dummy item when no results to prevent crash
                <DialogComboboxOptionListSelectItem
                  value=""
                  onChange={() => {}}
                  checked={false}
                  disabled
                >
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
