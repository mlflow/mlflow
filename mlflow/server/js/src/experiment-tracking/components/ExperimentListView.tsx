import React, { Component } from 'react';
import { Theme } from '@emotion/react';
import {
  WithDesignSystemThemeHoc,
  DesignSystemHocProps,
  Button,
  Tooltip,
  TableFilterLayout,
  TableFilterInput,
  Spacer,
  Header,
  Popover,
  InfoIcon,
  Typography,
  Alert,
} from '@databricks/design-system';
import 'react-virtualized/styles.css';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { withRouterNext, WithRouterNextProps } from '../../common/utils/withRouterNext';
import { ExperimentEntity } from '../types';
import { defaultContext, QueryClient } from '../../common/utils/reactQueryHooks';
import { ExperimentListQueryKeyHeader, useExperimentListQuery } from './experiment-page/hooks/useExperimentListQuery';
import { RowSelectionState, Updater } from '@tanstack/react-table';
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { ExperimentSearchSyntaxDocUrl } from '../../common/constants';
import { ExperimentListTable } from './ExperimentViewTable';

type Props = {
  experiments: ExperimentEntity[];
  pagination: Pick<
    ReturnType<typeof useExperimentListQuery>,
    'hasNextPage' | 'hasPreviousPage' | 'onNextPage' | 'onPreviousPage' | 'isLoading' | 'error'
  >;
} & WithRouterNextProps &
  DesignSystemHocProps;

type State = {
  checkedKeys: string[];
  searchInput: string;
  showCreateExperimentModal: boolean;
};

export class ExperimentListView extends Component<Props, State> {
  static contextType = defaultContext;
  // declare context: QueryClient; // FIXME

  invalidateExperimentList = () => {
    this.context.invalidateQueries({ queryKey: [ExperimentListQueryKeyHeader] });
  };

  state = {
    checkedKeys: [] as string[],
    searchInput: '',
    showCreateExperimentModal: false,
  };

  filterExperiments = (searchInput: string) => {
    const { experiments } = this.props;
    const lowerCasedSearchInput = searchInput.toLowerCase();
    return lowerCasedSearchInput === ''
      ? this.props.experiments
      : experiments.filter(({ name }) => name.toLowerCase().includes(lowerCasedSearchInput));
  };

  handleSearchInputChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    this.setState({
      searchInput: event.target.value,
    });
  };

  handleCreateExperiment = () => {
    this.setState({
      showCreateExperimentModal: true,
    });
  };

  handleCloseCreateExperimentModal = () => {
    this.setState({
      showCreateExperimentModal: false,
    });
  };

  pushExperimentRoute = () => {
    if (this.state.checkedKeys.length > 0) {
      const route =
        this.state.checkedKeys.length === 1
          ? Routes.getExperimentPageRoute(this.state.checkedKeys[0])
          : Routes.getCompareExperimentsPageRoute(this.state.checkedKeys);
      this.props.navigate(route);
    }
  };

  getSelectedRows = () => {
    return Object.fromEntries(
      this.props.experiments.map((experiment) => [
        experiment.experimentId,
        this.state.checkedKeys.includes(experiment.experimentId),
      ]),
    );
  };

  setSelectedRows = (updater: Updater<RowSelectionState>) => {
    const rowSelection = this.getSelectedRows();
    const newRowSelection = typeof updater === 'function' ? updater(rowSelection) : updater;

    this.setState({
      checkedKeys: Object.entries(newRowSelection)
        .filter(([_, value]) => value)
        .map(([key, _]) => key),
    });
  };

  render() {
    const { pagination } = this.props;
    const { error, isLoading, onNextPage, onPreviousPage, hasNextPage, hasPreviousPage } = pagination;

    const { searchInput } = this.state;
    const filteredExperiments = this.filterExperiments(searchInput);

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
                onClick={this.handleCreateExperiment}
                data-testid="create-experiment-button"
              >
                Create experiment
              </Button>
              <Tooltip
                componentId="mlflow.experiment_list_view.compare_experiments_button"
                content="Select at least two experiments from the table to compare them"
              >
                <Button
                  componentId="mlflow.experiment_list_view.new_experiment_button"
                  onClick={this.pushExperimentRoute}
                  data-testid="create-experiment-button"
                  disabled={this.state.checkedKeys.length < 2}
                >
                  Compare experiments
                </Button>
              </Tooltip>
            </>
          }
        />
        <Spacer shrinks={false} />
        {error && (
          <>
            <Alert
              type="error"
              message={error.message || 'A network error occurred.'}
              componentId="mlflow.experiment_list_view.error"
              closable={false}
            />
            <Spacer />
          </>
        )}
        <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <TableFilterLayout>
            <TableFilterInput
              placeholder="Search experiments by name"
              componentId="mlflow.experiment_list_view.search"
              value={searchInput}
              onChange={this.handleSearchInputChange}
              suffix={<ModelSearchInputHelpTooltip />}
            />
          </TableFilterLayout>
          <ExperimentListTable
            experiments={filteredExperiments}
            error={error}
            hasNextPage={hasNextPage}
            hasPreviousPage={hasPreviousPage}
            isLoading={isLoading}
            isFiltered={Boolean(searchInput)}
            onNextPage={onNextPage}
            onPreviousPage={onPreviousPage}
            rowSelection={this.getSelectedRows()}
            setRowSelection={this.setSelectedRows}
          />
        </div>
        <CreateExperimentModal
          isOpen={this.state.showCreateExperimentModal}
          onClose={this.handleCloseCreateExperimentModal}
          onExperimentCreated={this.invalidateExperimentList}
        />
      </ScrollablePageWrapper>
    );
  }
}

export default withRouterNext(WithDesignSystemThemeHoc(ExperimentListView));

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
              <Typography.Text>"attributes.name = 'x'" # or "name = 'x'"</Typography.Text>
            </li>
            <li>
              <Typography.Text>"attributes.name LIKE 'x%'"</Typography.Text>
            </li>
            <li>
              <Typography.Text>"tags.group != 'x'"</Typography.Text>
            </li>
            <li>
              <Typography.Text>"tags.group ILIKE '%x%'"</Typography.Text>
            </li>
            <li>
              <Typography.Text>"attributes.name LIKE 'x%' AND tags.group = 'y'"</Typography.Text>
            </li>
          </ul>
        </div>
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
};
