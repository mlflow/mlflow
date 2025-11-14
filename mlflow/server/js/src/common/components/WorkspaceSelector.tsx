import React, { useEffect, useMemo, useState } from 'react';
import { Button, DropdownMenu, Input, Tooltip, useDesignSystemTheme } from '@databricks/design-system';

import { shouldEnableWorkspaces } from '../utils/FeatureUtils';
import { fetchAPI, getAjaxUrl } from '../utils/FetchUtils';
import {
  DEFAULT_WORKSPACE_NAME,
  extractWorkspaceFromPathname,
  setActiveWorkspace,
  setAvailableWorkspaces,
} from '../utils/WorkspaceUtils';
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
  const [searchTerm, setSearchTerm] = useState('');
  const location = useLocation();
  const navigate = useNavigate();
  const { theme } = useDesignSystemTheme();

  const workspaceFromPath = extractWorkspaceFromPathname(location.pathname);
  const currentWorkspace = workspaceFromPath ?? DEFAULT_WORKSPACE_NAME;

  useEffect(() => {
    if (!workspacesEnabled) {
      setWorkspaces([]);
      setLoadFailed(false);
      return;
    }

    let isMounted = true;
    const loadWorkspaces = async () => {
      setLoading(true);
      setLoadFailed(false);
      try {
        const response = await fetchAPI(getAjaxUrl(WORKSPACES_ENDPOINT));
        if (!isMounted) {
          return;
        }

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
        setAvailableWorkspaces(filteredWorkspaces.map((w) => w.name));
      } catch {
        if (isMounted) {
          setLoadFailed(true);
          setAvailableWorkspaces([]);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    loadWorkspaces().catch(() => undefined);

    return () => {
      isMounted = false;
    };
  }, [workspacesEnabled]);

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
    if (!searchTerm) {
      return options;
    }
    const lowerSearch = searchTerm.toLowerCase();
    return options.filter((workspace) => workspace.name.toLowerCase().includes(lowerSearch));
  }, [options, searchTerm]);

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
    setSearchTerm(''); // Clear search on selection
  };

  if (!workspacesEnabled) {
    return null;
  }

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button componentId="workspace_selector" loading={loading} type="tertiary">
          {currentWorkspace}
        </Button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Content
        align="start"
        css={{
          minWidth: 250,
          maxHeight: 400,
          overflowY: 'auto',
          zIndex: 9999, // Ensure dropdown appears above all other content
        }}
      >
        <div
          css={{
            padding: theme.spacing.sm,
            position: 'sticky',
            top: 0,
            backgroundColor: theme.colors.backgroundPrimary,
            zIndex: 1,
          }}
          onClick={(e) => e.stopPropagation()} // Prevent dropdown from closing when clicking input
        >
          <Input
            placeholder="Filter workspaces..."
            value={searchTerm}
            onChange={(e) => {
              e.stopPropagation(); // Prevent event bubbling
              setSearchTerm(e.target.value);
            }}
            onClick={(e) => e.stopPropagation()} // Prevent dropdown from closing
            componentId="workspace_filter"
            autoFocus
          />
        </div>

        {loadFailed && (
          <DropdownMenu.Label css={{ color: theme.colors.textValidationDanger }}>
            Failed to load workspaces
          </DropdownMenu.Label>
        )}

        {filteredOptions.length === 0 && searchTerm && <DropdownMenu.Label>No workspaces found</DropdownMenu.Label>}

        {filteredOptions.map((workspace) => (
          <Tooltip
            key={workspace.name}
            content={workspace.description || workspace.name}
            componentId={`workspace_tooltip_${workspace.name}`}
          >
            <DropdownMenu.Item
              componentId={`workspace_item_${workspace.name}`}
              onClick={() => handleWorkspaceChange(workspace.name)}
              css={{
                fontWeight: workspace.name === currentWorkspace ? 'bold' : 'normal',
              }}
            >
              {workspace.name}
            </DropdownMenu.Item>
          </Tooltip>
        ))}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
