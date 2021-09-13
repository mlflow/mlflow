import React from 'react';
import PropTypes from 'prop-types';
import { Table, Input, Alert, Form } from 'antd';
import { Link } from 'react-router-dom';
import './ModelListView.css';
import { getModelPageRoute, getModelVersionPageRoute } from '../routes';
import Utils from '../../common/utils/Utils';
import {
  AntdTableSortOrder,
  Stages,
  StageTagComponents,
  EMPTY_CELL_PLACEHOLDER,
  REGISTERED_MODELS_PER_PAGE,
  REGISTERED_MODELS_SEARCH_NAME_FIELD,
  REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD,
} from '../constants';
import {
  ModelRegistryDocUrl,
  ModelRegistryOnboardingString,
  onboarding,
} from '../../common/constants';
import { SimplePagination } from '../../common/components/SimplePagination';
import { Spinner } from '../../common/components/Spinner';
import { CreateModelButton } from './CreateModelButton';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { css } from 'emotion';
import { CollapsibleTagsCell } from '../../common/components/CollapsibleTagsCell';
import { RegisteredModelTag } from '../sdk/ModelRegistryMessages';
import filterIcon from '../../common/static/filter-icon.svg';
import { CSSTransition } from 'react-transition-group';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import { FlexBar } from '../../shared/building_blocks/FlexBar';
import { Button } from '../../shared/building_blocks/Button';
import { Spacer } from '../../shared/building_blocks/Spacer';
import { SearchBox } from '../../shared/building_blocks/SearchBox';
import { mlPagePadding } from '../../shared/styleConstants';
import { FormattedMessage, injectIntl } from 'react-intl';

const NAME_COLUMN_INDEX = 'name';
const LAST_MODIFIED_COLUMN_INDEX = 'last_updated_timestamp';

const getOverallLatestVersionNumber = (latest_versions) =>
  latest_versions && Math.max(...latest_versions.map((v) => v.version));

const getLatestVersionNumberByStage = (latest_versions, stage) => {
  const modelVersion = latest_versions && latest_versions.find((v) => v.current_stage === stage);
  return modelVersion && modelVersion.version;
};

export class ModelListViewImpl extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: false,
      lastNavigationActionWasClickPrev: false,
      maxResultsSelection: REGISTERED_MODELS_PER_PAGE,
      showOnboardingHelper: this.showOnboardingHelper(),
      showFilters: false,
      nameSearchInput: props.nameSearchInput,
      tagSearchInput: props.tagSearchInput,
    };
  }

  static propTypes = {
    models: PropTypes.array.isRequired,
    nameSearchInput: PropTypes.string.isRequired,
    tagSearchInput: PropTypes.string.isRequired,
    orderByKey: PropTypes.string.isRequired,
    orderByAsc: PropTypes.bool.isRequired,
    currentPage: PropTypes.number.isRequired,
    // To know if there is a next page. If null, there is no next page. If undefined, we haven't
    // gotten an answer from the backend yet.
    nextPageToken: PropTypes.string,
    loading: PropTypes.bool,
    onSearch: PropTypes.func.isRequired,
    onClear: PropTypes.func.isRequired,
    onClickNext: PropTypes.func.isRequired,
    onClickPrev: PropTypes.func.isRequired,
    onClickSortableColumn: PropTypes.func.isRequired,
    onSetMaxResult: PropTypes.func.isRequired,
    getMaxResultValue: PropTypes.func.isRequired,
    intl: PropTypes.any,
  };

  static defaultProps = {
    models: [],
    nameSearchInput: '',
    tagSearchInput: '',
  };

  showOnboardingHelper() {
    const onboardingInformationStore = ModelListViewImpl.getLocalStore(onboarding);
    return onboardingInformationStore.getItem('showRegistryHelper') === null;
  }

  disableOnboardingHelper() {
    const onboardingInformationStore = ModelListViewImpl.getLocalStore(onboarding);
    onboardingInformationStore.setItem('showRegistryHelper', 'false');
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ModelRegistry component.
   */
  static getLocalStore(key) {
    return LocalStorageUtils.getStoreForComponent('ModelListView', key);
  }

  componentDidMount() {
    const pageTitle = 'MLflow Models';
    Utils.updatePageTitle(pageTitle);
  }

  renderModelVersionLink(name, versionNumber) {
    return (
      <FormattedMessage
        defaultMessage='<link>Version {versionNumber}</link>'
        description='Row entry for version columns in the registered model page'
        values={{
          versionNumber: versionNumber,
          link: (chunks) => (
            <Link to={getModelVersionPageRoute(name, versionNumber)}>{chunks}</Link>
          ),
        }}
      />
    );
  }

  getSortOrder = (key) => {
    const { orderByKey, orderByAsc } = this.props;
    if (key !== orderByKey) {
      return null;
    }
    return { sortOrder: orderByAsc ? AntdTableSortOrder.ASC : AntdTableSortOrder.DESC };
  };

  handleCellToggle = () => {
    this.forceUpdate();
  };

  getColumns = () => {
    const columns = [
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Name',
          description: 'Column title for model name in the registered model page',
        }),
        className: 'model-name',
        dataIndex: NAME_COLUMN_INDEX,
        render: (text, row) => {
          return <Link to={getModelPageRoute(row.name)}>{text}</Link>;
        },
        sorter: true,
        ...this.getSortOrder(REGISTERED_MODELS_SEARCH_NAME_FIELD),
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Latest Version',
          description: 'Column title for latest model version in the registered model page',
        }),
        className: 'latest-version',
        render: ({ name, latest_versions }) => {
          const versionNumber = getOverallLatestVersionNumber(latest_versions);
          return versionNumber
            ? this.renderModelVersionLink(name, versionNumber)
            : EMPTY_CELL_PLACEHOLDER;
        },
      },
      {
        title: StageTagComponents[Stages.STAGING],
        className: 'latest-staging',
        render: ({ name, latest_versions }) => {
          const versionNumber = getLatestVersionNumberByStage(latest_versions, Stages.STAGING);
          return versionNumber
            ? this.renderModelVersionLink(name, versionNumber)
            : EMPTY_CELL_PLACEHOLDER;
        },
      },
      {
        title: StageTagComponents[Stages.PRODUCTION],
        className: 'latest-production',
        render: ({ name, latest_versions }) => {
          const versionNumber = getLatestVersionNumberByStage(latest_versions, Stages.PRODUCTION);
          return versionNumber
            ? this.renderModelVersionLink(name, versionNumber)
            : EMPTY_CELL_PLACEHOLDER;
        },
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Last Modified',
          description:
            'Column title for last modified timestamp for a model in the registered model page',
        }),
        className: 'last-modified',
        dataIndex: LAST_MODIFIED_COLUMN_INDEX,
        render: (text, row) => <span>{Utils.formatTimestamp(row.last_updated_timestamp)}</span>,
        sorter: true,
        ...this.getSortOrder(REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD),
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Tags',
          description: 'Column title for model tags in the registered model page',
        }),
        className: 'table-tag-container',
        render: (row, index) => {
          return index.tags && index.tags.length > 0 ? (
            <div style={{ wordWrap: 'break-word', wordBreak: 'break-word' }}>
              <CollapsibleTagsCell
                tags={{ ...index.tags.map((tag) => RegisteredModelTag.fromJs(tag)) }}
                onToggle={this.handleCellToggle}
              />
            </div>
          ) : (
            EMPTY_CELL_PLACEHOLDER
          );
        },
      },
    ];
    return columns;
  };

  getRowKey = (record) => record.name;

  setLoadingFalse = () => {
    this.setState({ loading: false });
  };

  handleSearch = (event) => {
    event.preventDefault();
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onSearch(
      this.state.nameSearchInput,
      this.state.tagSearchInput,
      this.setLoadingFalse,
      this.setLoadingFalse,
    );
  };

  static getSortFieldName = (column) => {
    switch (column) {
      case NAME_COLUMN_INDEX:
        return REGISTERED_MODELS_SEARCH_NAME_FIELD;
      case LAST_MODIFIED_COLUMN_INDEX:
        return REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD;
      default:
        return null;
    }
  };

  handleTableChange = (pagination, filters, sorter) => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onClickSortableColumn(
      ModelListViewImpl.getSortFieldName(sorter.field),
      sorter.order,
      this.setLoadingFalse,
      this.setLoadingFalse,
    );
  };

  renderOnboardingContent() {
    const learnMoreLinkUrl = ModelListViewImpl.getLearnMoreLinkUrl();
    const learnMoreDisplayString = ModelListViewImpl.getLearnMoreDisplayString();
    const content = (
      <div>
        {learnMoreDisplayString}{' '}
        <FormattedMessage
          defaultMessage='<link>Learn more</link>'
          description='Learn more link on the model list page with cloud-specific link'
          values={{
            link: (chunks) => (
              <a
                href={learnMoreLinkUrl}
                target='_blank'
                rel='noopener noreferrer'
                className='LinkColor'
              >
                {chunks}
              </a>
            ),
          }}
        />
      </div>
    );

    return this.state.showOnboardingHelper ? (
      <Alert
        className='onboarding-information'
        description={content}
        type='info'
        showIcon
        closable
        onClose={() => this.disableOnboardingHelper()}
      />
    ) : null;
  }

  getEmptyTextComponent() {
    const { nameSearchInput, tagSearchInput } = this.props;
    const { lastNavigationActionWasClickPrev } = this.state;
    // Handle the case when emptiness is caused by search filter
    if (nameSearchInput || tagSearchInput) {
      if (lastNavigationActionWasClickPrev) {
        return (
          'No models found for the page. Please refresh the page as the underlying data may ' +
          'have changed significantly.'
        );
      } else {
        return 'No models found.';
      }
    }
    return (
      <div>
        <span>
          <FormattedMessage
            defaultMessage='No models yet. <link>Create a model</link> to get started.'
            description='Placeholder text for empty models table in the registered model list page'
            values={{
              link: (chunks) => <CreateModelButton buttonType='link' buttonText={chunks} />,
            }}
          />
        </span>
      </div>
    );
  }

  static getLearnMoreLinkUrl = () => ModelRegistryDocUrl;

  static getLearnMoreDisplayString = () => ModelRegistryOnboardingString;

  handleClickNext = () => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onClickNext(this.setLoadingFalse, this.setLoadingFalse);
  };

  handleClickPrev = () => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: true });
    this.props.onClickPrev(this.setLoadingFalse, this.setLoadingFalse);
  };

  handleSetMaxResult = ({ item, key, keyPath, domEvent }) => {
    this.setState({ loading: true });
    this.props.onSetMaxResult(key, this.setLoadingFalse, this.setLoadingFalse);
  };

  handleFilterToggle = () => {
    this.setState((previousState) => ({ showFilters: !previousState.showFilters }));
  };

  handleNameSearchInput = (event) => {
    this.setState({ nameSearchInput: event.target.value });
  };

  handleTagSearchInput = (event) => {
    this.setState({ tagSearchInput: event.target.value });
  };

  handleClear = () => {
    this.setState({ nameSearchInput: '', tagSearchInput: '' });
    this.props.onClear(this.setLoadingFalse, this.setLoadingFalse);
  };

  render() {
    const { models, currentPage, nextPageToken } = this.props;
    const { loading } = this.state;

    const title = (
      <Spacer size='small' direction='horizontal'>
        <span>
          <FormattedMessage
            defaultMessage='Registered Models'
            description='Header for displaying models in the model registry'
          />
        </span>
      </Spacer>
    );
    return (
      <div data-test-id='ModelListView-container' className={styles.rootContainer}>
        <PageHeader title={title} />
        {this.renderOnboardingContent()}
        <FlexBar
          left={
            <Spacer size='small' direction='horizontal'>
              <span className={`${styles.createModelButtonWrapper}`}>
                <CreateModelButton />
              </span>
            </Spacer>
          }
          right={
            <Spacer direction='horizontal' size='small'>
              <div className={styles.nameSearchBox}>
                <SearchBox
                  onChange={this.handleNameSearchInput}
                  value={this.state.nameSearchInput}
                  onSearch={this.handleSearch}
                  onPressEnter={this.handleSearch}
                  placeholder={this.props.intl.formatMessage({
                    defaultMessage: 'Search by model name',
                    description: 'Placeholder text inside model search bar',
                  })}
                />
              </div>
              <Button dataTestId='filter-button' onClick={this.handleFilterToggle}>
                <img className='filterIcon' src={filterIcon} alt='Filter' />
                <FormattedMessage
                  defaultMessage='Filter'
                  description='String for the filter button to filter model registry table
                   for models'
                />
              </Button>
              <Button dataTestId='clear-button' onClick={this.handleClear}>
                <FormattedMessage
                  defaultMessage='Clear'
                  description='String for the clear button to clear the text for searching models'
                />
              </Button>
            </Spacer>
          }
        />
        <div className='ModelListView-filter-dropdown'>
          <CSSTransition
            in={this.state.showFilters}
            timeout={300}
            classNames='lifecycleButtons'
            unmountOnExit
          >
            <FlexBar
              left={<div />}
              right={
                <Form.Item className={styles.tagLabelWrapper} label='Tags' labelCol={{ span: 24 }}>
                  <Input
                    name='tags-search'
                    data-testid='ModelListView-tagSearchBox'
                    aria-label='Search Tags'
                    type='text'
                    placeholder={`Search tags: tags.key='value'`}
                    value={this.state.tagSearchInput}
                    onChange={this.handleTagSearchInput}
                    onPressEnter={this.handleSearch}
                  />
                </Form.Item>
              }
            />
          </CSSTransition>
        </div>
        <Table
          size='middle'
          rowKey={this.getRowKey}
          className='model-version-table'
          dataSource={models}
          columns={this.getColumns()}
          locale={{ emptyText: this.getEmptyTextComponent() }}
          pagination={{
            hideOnSinglePage: true,
            pageSize: this.props.getMaxResultValue(),
          }}
          loading={loading && { indicator: <Spinner /> }}
          onChange={this.handleTableChange}
        />
        <div>
          <SimplePagination
            currentPage={currentPage}
            loading={this.props.loading}
            isLastPage={nextPageToken === null}
            onClickNext={this.handleClickNext}
            onClickPrev={this.handleClickPrev}
            handleSetMaxResult={this.handleSetMaxResult}
            maxResultOptions={[REGISTERED_MODELS_PER_PAGE, 25, 50, 100]}
            getSelectedPerPageSelection={this.props.getMaxResultValue}
          />
        </div>
      </div>
    );
  }
}

export const ModelListView = injectIntl(ModelListViewImpl);

const styles = {
  tagLabelWrapper: css({
    paddingBottom: '0',
    paddingTop: '16px',
    width: '614px',
  }),
  createModelButtonWrapper: css({
    marginLeft: 'auto',
    order: 2,
    height: '40px',
    width: '120px',
  }),
  nameSearchBox: css({
    width: '446px',
  }),
  rootContainer: css({
    margin: mlPagePadding,
  }),
};
