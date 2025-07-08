import React, { useState } from 'react';
import { Theme } from '@emotion/react';
import {
  Button,
  TableFilterLayout,
  TableFilterInput,
  Spacer,
  Header,
  Popover,
  InfoIcon,
  Typography,
  Alert,
  useDesignSystemTheme,
} from '@databricks/design-system';
import 'react-virtualized/styles.css';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { useExperimentListQuery, useInvalidateExperimentList } from './experiment-page/hooks/useExperimentListQuery';
import { RowSelectionState } from '@tanstack/react-table';
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { ExperimentSearchSyntaxDocUrl } from '../../common/constants';
import { ExperimentListTable } from './ExperimentListTable';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { BulkDeleteExperimentModal } from './modals/BulkDeleteExperimentModal';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';

type Props = {
  searchFilter: string;
  setSearchFilter: (searchFilter: string) => void;
};

export const ExperimentListView = ({ searchFilter, setSearchFilter }: Props) => {
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
              defaultMessage: 'Filter experiments by name, tags or attributes',
              description: 'Placeholder text inside experiments search bar',
            })}
            componentId="mlflow.experiment_list_view.search"
            defaultValue={searchFilter}
            onChange={handleSearchInputChange}
            onSubmit={handleSearchSubmit}
            onClear={handleSearchClear}
            showSearchButton
            suffix={<ModelSearchInputHelpTooltip />}
          />
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
    </ScrollablePageWrapper>
  );
};

export default ExperimentListView;

const ModelSearchInputHelpTooltip = () => {
  const { formatMessage } = useIntl();
  const tooltipIntroMessage = defineMessage({
    defaultMessage:
      'A filter expression over experiment attributes and tags that allows returning a subset of experiments.',
    description: 'Tooltip string to explain how to search experiments',
  });

  // Tooltips are not expected to contain links.
  const labelText = formatMessage(tooltipIntroMessage, { newline: ' ', whereBold: 'WHERE' });

  return (
    <Popover.Root componentId="mlflow.experiment_list_view.searchbox.help_popover.root">
      <Popover.Trigger
        aria-label={labelText}
        css={{ border: 0, background: 'none', padding: 0, lineHeight: 0, cursor: 'pointer' }}
      >
        <InfoIcon />
      </Popover.Trigger>
      <Popover.Content align="start">
        <div>
          <FormattedMessage {...tooltipIntroMessage} />
          <Typography.Paragraph>
            The syntax is a subset of SQL that supports ANDing together binary operations between an attribute or tag,
            and a constant.
          </Typography.Paragraph>
          <Typography.Paragraph>
            <FormattedMessage
              defaultMessage="<link>Learn more</link>"
              description="Learn more tooltip link to learn more on how to search experiments"
              values={{
                link: (chunks) => (
                  <Typography.Link
                    componentId="mlflow.experiment_list_view.searchbox.help_popover.syntax_url"
                    href={ExperimentSearchSyntaxDocUrl + '#syntax'}
                    openInNewTab
                  >
                    {chunks}
                  </Typography.Link>
                ),
              }}
            />
          </Typography.Paragraph>
          <Typography.Paragraph>
            <FormattedMessage
              defaultMessage="Examples:"
              description="Text header for examples of mlflow search syntax"
            />
          </Typography.Paragraph>
          <ul>
            <li>
              <Typography.Text code>attributes.name = 'x'</Typography.Text>
              <Typography.Text> or </Typography.Text>
              <Typography.Text code>name = 'x'</Typography.Text>
            </li>
            <li>
              <Typography.Text code>attributes.name LIKE 'x%'</Typography.Text>
            </li>
            <li>
              <Typography.Text code>tags.group != 'x'</Typography.Text>
            </li>
            <li>
              <Typography.Text code>tags.group ILIKE '%x%'</Typography.Text>
            </li>
            <li>
              <Typography.Text code>attributes.name LIKE 'x%' AND tags.group = 'y'</Typography.Text>
            </li>
          </ul>
        </div>
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
};
