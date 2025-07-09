import React, { useState } from 'react';
import { Theme } from '@emotion/react';
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
  RHFControlledComponents,
  NoIcon,
  CloseIcon,
  PlusIcon,
} from '@databricks/design-system';
import 'react-virtualized/styles.css';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { useExperimentListQuery, useInvalidateExperimentList } from './experiment-page/hooks/useExperimentListQuery';
import { RowSelectionState } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { ExperimentListTable } from './ExperimentListTable';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { BulkDeleteExperimentModal } from './modals/BulkDeleteExperimentModal';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { useUpdateExperimentTags } from './experiment-page/hooks/useUpdateExperimentTags';
import { useFieldArray, useForm } from 'react-hook-form';
import { useSearchFilter } from './experiment-page/hooks/useSearchFilter';

export const ExperimentListView = () => {
  const [searchFilter, setSearchFilter] = useSearchFilter();
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
  } = useExperimentListQuery({ searchFilter });
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

  const pushExperimentRoute = () => {
    const route = Routes.getCompareExperimentsPageRoute(checkedKeys);
    navigate(route);
  };

  const checkedKeys = Object.entries(rowSelection)
    .filter(([_, value]) => value)
    .map(([key, _]) => key);

  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const intl = useIntl();

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
          <Popover.Root componentId="mlflow.experiment_list_view.tag_filter">
            <Popover.Trigger asChild>
              <Button
                componentId="mlflow.experiment_list_view.tag_filter.trigger"
                icon={<FilterIcon />}
                endIcon={<ChevronDownIcon />}
              >
                Tag filter
              </Button>
            </Popover.Trigger>
            <Popover.Content>
              <TagFilters />
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

const EMPTY_TAG = { key: '', value: '', operator: 'IS' } as const;
const OPERATORS = ['IS', 'IS NOT', 'CONTAINS'] as const;
type Operator = typeof OPERATORS[number];
type TagFilter = { key: string; value: string; operator: Operator };

function TagFilters() {
  const { control, handleSubmit } = useForm<{ tagFilters: TagFilter[] }>({
    defaultValues: { tagFilters: [EMPTY_TAG] },
  });
  const { fields, append, remove } = useFieldArray({ control, name: 'tagFilters' });
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  return (
    <form
      onSubmit={handleSubmit((data) => console.log(data))}
      css={{ display: 'flex', flexDirection: 'column', alignItems: 'end', padding: theme.spacing.md }}
    >
      <fieldset css={{ display: 'flex', flexDirection: 'column', alignItems: 'start', gap: theme.spacing.sm }}>
        {fields.map((field, index) => (
          <div key={field.id} css={{ display: 'flex', gap: theme.spacing.sm }}>
            <RHFControlledComponents.Input
              componentId=""
              name={`tagFilters.${index}.key`}
              control={control}
              // aria-label={
              //   valueRequired
              //     ? intl.formatMessage({
              //         defaultMessage: 'Value',
              //         description: 'Key-value tag editor modal > Value input label (required)',
              //       })
              //     : intl.formatMessage({
              //         defaultMessage: 'Value (optional)',
              //         description: 'Key-value tag editor modal > Value input label',
              //       })
              // }
              // placeholder={intl.formatMessage({
              //   defaultMessage: 'Type a value',
              //   description: 'Key-value tag editor modal > Value input placeholder',
              // })}
            />
            <RHFControlledComponents.LegacySelect
              name={`tagFilters.${index}.operator`}
              control={control}
              options={OPERATORS.map((op) => ({ key: op, value: op }))}
              css={{ minWidth: '14ch' }}
            />
            <RHFControlledComponents.Input componentId="" name={`tagFilters.${index}.value`} control={control} />
            <Button
              componentId=""
              type="tertiary"
              onClick={() => remove(index)}
              disabled={fields.length === 1}
              aria-label={formatMessage({
                defaultMessage: 'Remove filter',
                description: 'Button to remove a filter in the tags filter popover for experiments page search by tags',
              })}
            >
              <CloseIcon />
            </Button>
          </div>
        ))}
        <Button componentId="" onClick={() => append(EMPTY_TAG)} icon={<PlusIcon />}>
          Add filter
        </Button>
      </fieldset>
      <Button htmlType="submit" componentId="" type="primary">
        Apply filters
      </Button>
    </form>
  );
}
