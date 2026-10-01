import { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Empty,
  GridIcon,
  Header,
  ListIcon,
  LockIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  PuzzleIcon,
  Spacer,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { PermissionError } from '@databricks/web-shared/errors';
import { useDebounce } from 'use-debounce';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useSkillsListQuery } from '../hooks/useSkillsListQuery';
import { SkillCardGrid } from '../components/SkillCardGrid';
import { SkillListTable } from '../components/SkillListTable';
import { SkillListFilters } from '../components/SkillListFilters';
import { SkillRegistryBetaTag } from '../components/SkillRegistryBetaTag';
import { flexColumnContainerStyles, headerIconStyles } from '../styles';
import { hasSkillCatalogFilters } from '../utils';
import type { SkillSourceType } from '../types';

type ViewMode = 'list' | 'grid';

const isPermissionDeniedError = (error: Error | undefined) =>
  error instanceof PermissionError || error?.name === 'PermissionError';

const SkillRegistryPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [searchFilter, setSearchFilter] = useState('');
  const [filterActive, setFilterActive] = useState(false);
  const [organization, setOrganization] = useState('');
  const [sourceType, setSourceType] = useState<SkillSourceType | ''>('');
  const [debouncedSearchFilter] = useDebounce(searchFilter, 500);
  const seenOrganizations = useRef(new Set<string>());
  const [organizations, setOrganizations] = useState<string[]>([]);

  const catalogFilters = {
    searchText: debouncedSearchFilter,
    filterActive,
    organization,
    sourceType,
  };

  const {
    data: skills,
    isLoading,
    error,
    hasNextPage,
    hasPreviousPage,
    onNextPage,
    onPreviousPage,
    pageSizeSelect,
    refetch,
  } = useSkillsListQuery(catalogFilters);

  const hasActiveFilters = hasSkillCatalogFilters(catalogFilters);
  const isPermissionDenied = isPermissionDeniedError(error);

  useEffect(() => {
    let added = false;
    for (const skill of skills ?? []) {
      if (skill.organization && !seenOrganizations.current.has(skill.organization)) {
        seenOrganizations.current.add(skill.organization);
        added = true;
      }
    }
    if (added) {
      setOrganizations(Array.from(seenOrganizations.current).sort((left, right) => left.localeCompare(right)));
    }
  }, [skills]);

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span css={headerIconStyles(theme)}>
              <PuzzleIcon />
            </span>
            <FormattedMessage defaultMessage="Skills" description="Skill Registry page title" />
            <SkillRegistryBetaTag />
          </span>
        }
      />
      <Spacer shrinks={false} />
      <div css={flexColumnContainerStyles}>
        <div css={{ flexShrink: 0, width: '100%' }}>
          <SkillListFilters
            searchFilter={searchFilter}
            onSearchFilterChange={setSearchFilter}
            filterActive={filterActive}
            onFilterActiveChange={setFilterActive}
            organization={organization}
            onOrganizationChange={setOrganization}
            organizations={organizations}
            sourceType={sourceType}
            onSourceTypeChange={setSourceType}
            actions={
              <SegmentedControlGroup
                name="skill-registry-view-mode"
                value={viewMode}
                onChange={(e) => setViewMode(e.target.value as ViewMode)}
                componentId="mlflow.skill_registry.view_toggle"
              >
                <SegmentedControlButton
                  value="list"
                  icon={<ListIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'List view',
                    description: 'Aria label for Skill Registry list view toggle',
                  })}
                />
                <SegmentedControlButton
                  value="grid"
                  icon={<GridIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'Grid view',
                    description: 'Aria label for Skill Registry grid view toggle',
                  })}
                />
              </SegmentedControlGroup>
            }
          />
        </div>
        {isPermissionDenied ? (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: 400,
              width: '100%',
            }}
          >
            <Empty
              image={<LockIcon />}
              title={
                <FormattedMessage
                  defaultMessage="Permission denied"
                  description="Title for Skill Registry permission-denied state"
                />
              }
              description={
                error?.message || (
                  <FormattedMessage
                    defaultMessage="You do not have permission to view skills."
                    description="Description for Skill Registry permission-denied state"
                  />
                )
              }
            />
          </div>
        ) : (
          <>
            {error?.message && (
              <Alert
                type="error"
                message={error.message}
                componentId="mlflow.skill_registry.error"
                closable={false}
                css={{ marginTop: theme.spacing.sm, flexShrink: 0 }}
                actions={[
                  {
                    componentId: 'mlflow.skill_registry.error.retry',
                    children: (
                      <FormattedMessage
                        defaultMessage="Retry"
                        description="Retry button for Skill Registry catalog error"
                      />
                    ),
                    onClick: () => refetch(),
                  },
                ]}
              />
            )}
            {!error &&
              (viewMode === 'grid' ? (
                <SkillCardGrid
                  skills={skills}
                  isLoading={isLoading}
                  isFiltered={hasActiveFilters}
                  hasNextPage={hasNextPage}
                  hasPreviousPage={hasPreviousPage}
                  onNextPage={onNextPage}
                  onPreviousPage={onPreviousPage}
                  pageSizeSelect={pageSizeSelect}
                />
              ) : (
                <SkillListTable
                  skills={skills}
                  hasNextPage={hasNextPage}
                  hasPreviousPage={hasPreviousPage}
                  isLoading={isLoading}
                  isFiltered={hasActiveFilters}
                  onNextPage={onNextPage}
                  onPreviousPage={onPreviousPage}
                  pageSizeSelect={pageSizeSelect}
                />
              ))}
          </>
        )}
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.SKILL_REGISTRY, SkillRegistryPage);
