import React, { useState } from 'react';
import { Interpolation, Theme } from '@emotion/react';
import {
  Button,
  TableFilterLayout,
  TableFilterInput,
  Spacer,
  Header,
  Alert,
  useDesignSystemTheme,
  Popover,
  FilterIcon,
  ChevronDownIcon,
} from '@databricks/design-system';
import 'react-virtualized/styles.css';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { useExperimentListQuery, useInvalidateExperimentList } from './experiment-page/hooks/useExperimentListQuery';
import type { RowSelectionState } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { ExperimentListTable } from './ExperimentListTable';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { BulkDeleteExperimentModal } from './modals/BulkDeleteExperimentModal';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { useUpdateExperimentTags } from './experiment-page/hooks/useUpdateExperimentTags';
import { useSearchFilter } from './experiment-page/hooks/useSearchFilter';
import { TagFilter, useTagsFilter } from './experiment-page/hooks/useTagsFilter';
import { ExperimentListViewTagsFilter } from './experiment-page/components/ExperimentListViewTagsFilter';

export const ExperimentListView = () => {
  const [searchFilter, setSearchFilter] = useSearchFilter();
  const { tagsFilter, setTagsFilter, isTagsFilterOpen, setIsTagsFilterOpen } = useTagsFilter();

  const {
    data: experiments,
    isLoading,
    error,
    hasNextPage,
    hasPreviousPage,
    onNextPage,
    onPreviousPage,
    pageSizeSelect,
    sorting,
    setSorting,
  } = useExperimentListQuery({ searchFilter, tagsFilter });
  const invalidateExperimentList = useInvalidateExperimentList();

  const { EditTagsModal, showEditExperimentTagsModal } = useUpdateExperimentTags({
    onSuccess: invalidateExperimentList,
  });

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const [searchInput, setSearchInput] = useState('');
  const [showCreateExperimentModal, setShowCreateExperimentModal] = useState(false);
  const [showBulkDeleteExperimentModal, setShowBulkDeleteExperimentModal] = useState(false);

  const handleSearchInputChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    setSearchInput(event.target.value);
  };

  const handleSearchSubmit = () => {
    setSearchFilter(searchInput);
  };

  const handleSearchClear = () => {
    setSearchFilter('');
  };

  const handleCreateExperiment = () => {
    setShowCreateExperimentModal(true);
  };

  const handleCloseCreateExperimentModal = () => {
    setShowCreateExperimentModal(false);
  };

  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const intl = useIntl();

  const checkedKeys = Object.entries(rowSelection)
    .filter(([_, value]) => value)
    .map(([key, _]) => key);

  const pushExperimentRoute = () => {
    const route = Routes.getCompareExperimentsPageRoute(checkedKeys);
    navigate(route);
  };

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={<FormattedMessage defaultMessage="Experiments" description="Header title for the experiments page" />}
        buttons={
          <>
            <Button
              componentId="mlflow.experiment_list_view.new_experiment_button"
              type="primary"
              onClick={handleCreateExperiment}
              data-testid="create-experiment-button"
            >
              <FormattedMessage
                defaultMessage="Create"
                description="Label for the create experiment action on the experiments list page"
              />
            </Button>
            <Button
              componentId="mlflow.experiment_list_view.compare_experiments_button"
              onClick={pushExperimentRoute}
              data-testid="compare-experiment-button"
              disabled={checkedKeys.length < 2}
            >
              <FormattedMessage
                defaultMessage="Compare"
                description="Label for the compare experiments action on the experiments list page"
              />
            </Button>
            <Button
              componentId="mlflow.experiment_list_view.bulk_delete_button"
              onClick={() => setShowBulkDeleteExperimentModal(true)}
              data-testid="delete-experiments-button"
              disabled={checkedKeys.length < 1}
              danger
            >
              <FormattedMessage
                defaultMessage="Delete"
                description="Label for the delete experiments action on the experiments list page"
              />
            </Button>
          </>
        }
      />
      <Spacer shrinks={false} />
      {error && (
        <Alert
          css={{ marginBlockEnd: theme.spacing.sm }}
          type="error"
          message={
            error instanceof ErrorWrapper
              ? error.getMessageField()
              : error.message || (
                  <FormattedMessage
                    defaultMessage="A network error occurred."
                    description="Error message for generic network error"
                  />
                )
          }
          componentId="mlflow.experiment_list_view.error"
          closable={false}
        />
      )}
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <TableFilterLayout>
          <TableFilterInput
            data-testid="search-experiment-input"
            placeholder={intl.formatMessage({
              defaultMessage: 'Filter experiments by name',
              description: 'Placeholder text inside experiments search bar',
            })}
            componentId="mlflow.experiment_list_view.search"
            defaultValue={searchFilter}
            onChange={handleSearchInputChange}
            onSubmit={handleSearchSubmit}
            onClear={handleSearchClear}
            showSearchButton
          />
          <Popover.Root
            componentId="mlflow.experiment_list_view.tag_filter"
            open={isTagsFilterOpen}
            onOpenChange={setIsTagsFilterOpen}
          >
            <Popover.Trigger asChild>
              <Button
                componentId="mlflow.experiment_list_view.tag_filter.trigger"
                icon={<FilterIcon />}
                endIcon={<ChevronDownIcon />}
                type={tagsFilter.length > 0 ? 'primary' : undefined}
              >
                <FormattedMessage
                  defaultMessage="Tag filter"
                  description="Button to open the tags filter popover in the experiments page"
                />
              </Button>
            </Popover.Trigger>
            <Popover.Content>
              <ExperimentListViewTagsFilter tagsFilter={tagsFilter} setTagsFilter={setTagsFilter} />
            </Popover.Content>
          </Popover.Root>
        </TableFilterLayout>
        <ExperimentListTable
          experiments={experiments}
          isLoading={isLoading}
          isFiltered={Boolean(searchFilter)}
          rowSelection={rowSelection}
          setRowSelection={setRowSelection}
          cursorPaginationProps={{
            hasNextPage,
            hasPreviousPage,
            onNextPage,
            onPreviousPage,
            pageSizeSelect,
          }}
          sortingProps={{ sorting, setSorting }}
          onEditTags={showEditExperimentTagsModal}
        />
      </div>
      <CreateExperimentModal
        isOpen={showCreateExperimentModal}
        onClose={handleCloseCreateExperimentModal}
        onExperimentCreated={invalidateExperimentList}
      />
      <BulkDeleteExperimentModal
        experiments={(experiments ?? []).filter(({ experimentId }) => checkedKeys.includes(experimentId))}
        isOpen={showBulkDeleteExperimentModal}
        onClose={() => setShowBulkDeleteExperimentModal(false)}
        onExperimentsDeleted={() => {
          invalidateExperimentList();
          setRowSelection({});
        }}
      />
      {EditTagsModal}
    </ScrollablePageWrapper>
  );
};

export default ExperimentListView;
