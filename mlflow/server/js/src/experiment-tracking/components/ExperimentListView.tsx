import React, { useState } from 'react';
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
import { useInvalidateExperimentList } from './experiment-page/hooks/useExperimentListQuery';
import { RowSelectionState } from '@tanstack/react-table';
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { ExperimentSearchSyntaxDocUrl } from '../../common/constants';
import { ExperimentListTable } from './ExperimentListTable';

type Props = {
  experiments: ExperimentEntity[];
  error?: Error;
} & WithRouterNextProps &
  DesignSystemHocProps;

export const ExperimentListView = (props: Props) => {
  const invalidateExperimentList = useInvalidateExperimentList();

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const [searchInput, setSearchInput] = useState('');
  const [showCreateExperimentModal, setShowCreateExperimentModal] = useState(false);

  const filterExperiments = (searchInput: string) => {
    const { experiments } = props;
    const lowerCasedSearchInput = searchInput.toLowerCase();
    return lowerCasedSearchInput === ''
      ? props.experiments
      : experiments.filter(({ name }) => name.toLowerCase().includes(lowerCasedSearchInput));
  };

  const handleSearchInputChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    setSearchInput(event.target.value);
  };

  const handleCreateExperiment = () => {
    setShowCreateExperimentModal(true);
  };

  const handleCloseCreateExperimentModal = () => {
    setShowCreateExperimentModal(false);
  };

  const pushExperimentRoute = () => {
    const route = Routes.getCompareExperimentsPageRoute(checkedKeys);
    props.navigate(route);
  };

  const checkedKeys = Object.entries(rowSelection)
    .filter(([_, value]) => value)
    .map(([key, _]) => key);

  const { designSystemThemeApi, error } = props;
  const { theme } = designSystemThemeApi;

  const filteredExperiments = filterExperiments(searchInput);

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
                defaultMessage="Create experiment"
                description="Label for the create experiment action on the experiments list page"
              />
            </Button>
            <Tooltip
              componentId="mlflow.experiment_list_view.compare_experiments_button"
              content="Select at least two experiments from the table to compare them"
            >
              <Button
                componentId="mlflow.experiment_list_view.compare_experiment_button"
                onClick={pushExperimentRoute}
                data-testid="compare-experiment-button"
                disabled={checkedKeys.length < 2}
              >
                <FormattedMessage
                  defaultMessage="Compare experiments"
                  description="Label for the compare experiments action on the experiments list page"
                />
              </Button>
            </Tooltip>
          </>
        }
      />
      <Spacer shrinks={false} />
      {error && (
        <Alert
          css={{ marginBlockEnd: theme.spacing.sm }}
          type="error"
          message={error.message || 'A network error occurred.'}
          componentId="mlflow.experiment_list_view.error"
          closable={false}
        />
      )}
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <TableFilterLayout>
          <TableFilterInput
            placeholder="Search experiments by name"
            componentId="mlflow.experiment_list_view.search"
            value={searchInput}
            onChange={handleSearchInputChange}
            suffix={<ModelSearchInputHelpTooltip />}
          />
        </TableFilterLayout>
        <ExperimentListTable
          experiments={filteredExperiments}
          isFiltered={Boolean(searchInput)}
          rowSelection={rowSelection}
          setRowSelection={setRowSelection}
        />
      </div>
      <CreateExperimentModal
        isOpen={showCreateExperimentModal}
        onClose={handleCloseCreateExperimentModal}
        onExperimentCreated={invalidateExperimentList}
      />
    </ScrollablePageWrapper>
  );
};

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
