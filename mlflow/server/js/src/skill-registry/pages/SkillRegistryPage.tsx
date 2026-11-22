import { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Empty,
  Header,
  LockIcon,
  PlusIcon,
  PuzzleIcon,
  Spacer,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { PermissionError } from '@databricks/web-shared/errors';
import { useDebounce } from 'use-debounce';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useSkillsListQuery } from '../hooks/useSkillsListQuery';
import { SkillListTable } from '../components/SkillListTable';
import { SkillListFilters } from '../components/SkillListFilters';
import { SkillRegistryBetaTag } from '../components/SkillRegistryBetaTag';
import { RegisterSkillModal } from '../components/RegisterSkillModal';
import { flexColumnContainerStyles, headerIconStyles } from '../styles';
import SkillRegistryRoutes from '../routes';
import { hasSkillCatalogFilters } from '../utils';
import type { SkillSourceType } from '../types';

const isPermissionDeniedError = (error: Error | undefined) =>
  error instanceof PermissionError || error?.name === 'PermissionError';

const SkillRegistryPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const [registerOpen, setRegisterOpen] = useState(false);
  const [searchFilter, setSearchFilter] = useState('');
  const [filterActive, setFilterActive] = useState(true);
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
  const isPageEmpty = !isLoading && !error && !skills?.length && !hasPreviousPage;
  // Active is on by default, so an empty page with only that filter may still hide draft or deprecated skills.
  // Look once without it before treating the catalog as empty.
  const onlyActiveFilter = filterActive && !hasSkillCatalogFilters({ ...catalogFilters, filterActive: false });
  const {
    data: unfilteredSkills,
    isLoading: unfilteredLoading,
    error: unfilteredError,
  } = useSkillsListQuery({
    enabled: isPageEmpty && onlyActiveFilter,
  });
  // Keeps the list loading until that lookup answers, so "No skills found" doesn't flash before the create prompt.
  const isCheckingCatalog = isPageEmpty && onlyActiveFilter && unfilteredLoading;
  // A failed lookup proves nothing, so the page stays a filtered empty result rather than offering the first skill.
  const isCatalogEmpty =
    isPageEmpty &&
    !isCheckingCatalog &&
    (!hasActiveFilters || (onlyActiveFilter && !unfilteredError && !unfilteredSkills?.length));
  // Like the MCP registry, an empty catalog offers Create in its empty state instead. The header button waits
  // until the list shows there is something to add to, so it doesn't flash while the first page loads.
  const showHeaderCreate =
    Boolean(error) ||
    (hasActiveFilters && !onlyActiveFilter) ||
    hasPreviousPage ||
    Boolean(skills?.length) ||
    Boolean(unfilteredSkills?.length) ||
    Boolean(unfilteredError);
  const openRegister = () => setRegisterOpen(true);

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
        buttons={
          showHeaderCreate ? (
            <Button
              componentId="mlflow.skill_registry.create"
              type="primary"
              icon={<PlusIcon />}
              onClick={openRegister}
            >
              <FormattedMessage defaultMessage="Create skill" description="Button that opens skill registration" />
            </Button>
          ) : undefined
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
            {!error && (
              <SkillListTable
                skills={skills}
                hasNextPage={hasNextPage}
                hasPreviousPage={hasPreviousPage}
                isLoading={isLoading || isCheckingCatalog}
                isFiltered={hasActiveFilters && !isCatalogEmpty}
                onCreateSkill={isCatalogEmpty ? openRegister : undefined}
                onNextPage={onNextPage}
                onPreviousPage={onPreviousPage}
                pageSizeSelect={pageSizeSelect}
              />
            )}
          </>
        )}
      </div>
      <RegisterSkillModal
        visible={registerOpen}
        onClose={() => setRegisterOpen(false)}
        onRegistered={(version) =>
          navigate(SkillRegistryRoutes.getSkillDetailRoute(version.name, version.organization, version.version))
        }
      />
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.SKILL_REGISTRY, SkillRegistryPage);
