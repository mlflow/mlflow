/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import './ModelListView.css';
import Utils from '../../common/utils/Utils';
import {
  REGISTERED_MODELS_PER_PAGE_COMPACT,
  REGISTERED_MODELS_SEARCH_NAME_FIELD,
  REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD,
} from '../constants';
import { ModelRegistryDocUrl, ModelRegistryOnboardingString, onboarding } from '../../common/constants';
import { CreateModelButton } from './CreateModelButton';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { PageHeader } from '../../shared/building_blocks/PageHeader';

import { FormattedMessage, type IntlShape, injectIntl } from 'react-intl';
import { Alert, CursorPagination, Spacer as DuBoisSpacer, Spacer, Typography } from '@databricks/design-system';
import { shouldShowModelsNextUI } from '../../common/utils/FeatureUtils';
import { ModelListFilters } from './model-list/ModelListFilters';
import { ModelListTable } from './model-list/ModelListTable';
import { PageContainer } from '../../common/components/PageContainer';
import { ModelsNextUIToggleSwitch } from './ModelsNextUIToggleSwitch';
import { withNextModelsUIContext } from '../hooks/useNextModelsUI';

const NAME_COLUMN_INDEX = 'name';
const LAST_MODIFIED_COLUMN_INDEX = 'last_updated_timestamp';

type ModelListViewImplProps = {
  models: any[];
  showEditPermissionModal: (...args: any[]) => any;
  permissionLevel: string;
  selectedOwnerFilter: string;
  selectedStatusFilter: string;
  onOwnerFilterChange: (...args: any[]) => any;
  onStatusFilterChange: (...args: any[]) => any;
  searchInput: string;
  orderByKey: string;
  orderByAsc: boolean;
  currentPage: number;
  nextPageToken: string | null;
  loading?: boolean;
  error?: Error;
  onSearch: (...args: any[]) => any;
  onClickNext: (...args: any[]) => any;
  onClickPrev: (...args: any[]) => any;
  onClickSortableColumn: (...args: any[]) => any;
  onSetMaxResult: (...args: any[]) => any;
  maxResultValue: number;
  intl: IntlShape;
};

type ModelListViewImplState = any;

export class ModelListViewImpl extends React.Component<ModelListViewImplProps, ModelListViewImplState> {
  constructor(props: ModelListViewImplProps) {
    super(props);

    this.state = {
      maxResultsSelection: REGISTERED_MODELS_PER_PAGE_COMPACT,
    };
  }

  static defaultProps = {
    models: [],
    searchInput: '',
  };

  disableOnboardingHelper() {
    const onboardingInformationStore = ModelListViewImpl.getLocalStore(onboarding);
    onboardingInformationStore.setItem('showRegistryHelper', 'false');
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ModelRegistry component.
   */
  static getLocalStore(key: any) {
    return LocalStorageUtils.getStoreForComponent('ModelListView', key);
  }

  componentDidMount() {
    const pageTitle = 'MLflow Models';
    Utils.updatePageTitle(pageTitle);
  }

  handleSearch = (event: any, searchInput: any) => {
    event?.preventDefault();
    this.props.onSearch(searchInput);
  };

  static getSortFieldName = (column: any) => {
    switch (column) {
      case NAME_COLUMN_INDEX:
        return REGISTERED_MODELS_SEARCH_NAME_FIELD;
      case LAST_MODIFIED_COLUMN_INDEX:
        return REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD;
      default:
        return null;
    }
  };

  unifiedTableSortChange = ({ orderByKey, orderByAsc }: any) => {
    // Different column keys are used for sorting and data accessing,
    // mapping to proper keys happens below
    const fieldMappedToSortKey =
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      {
        timestamp: 'last_updated_timestamp',
      }[orderByKey] || orderByKey;

    this.handleTableChange(undefined, undefined, {
      field: fieldMappedToSortKey,
      order: orderByAsc ? 'undefined' : 'descend',
    });
  };

  handleTableChange = (pagination: any, filters: any, sorter: any) => {
    this.props.onClickSortableColumn(ModelListViewImpl.getSortFieldName(sorter.field), sorter.order);
  };

  static getLearnMoreLinkUrl = () => ModelRegistryDocUrl;

  static getLearnMoreDisplayString = () => ModelRegistryOnboardingString;

  handleClickNext = () => {
    this.props.onClickNext();
  };

  handleClickPrev = () => {
    this.props.onClickPrev();
  };

  handleSetMaxResult = ({ item, key, keyPath, domEvent }: any) => {
    this.props.onSetMaxResult(key);
  };

  render() {
    // prettier-ignore
    const {
      models,
      currentPage,
      nextPageToken,
      searchInput,
    } = this.props;
    const { loading, error } = this.props;

    // Determine if we use any filters at the moment
    const isFiltered =
      // prettier-ignore
      Boolean(searchInput);

    const title = (
      <FormattedMessage
        defaultMessage="Registered Models"
        description="Header for displaying models in the model registry"
      />
    );
    return (
      <PageContainer data-testid="ModelListView-container" usesFullHeight>
        <div>
          <PageHeader title={title} spacerSize="xs">
            <CreateModelButton />
          </PageHeader>
          {/* TODO[SHIP-6202]: Move the description to the Header prop 'description' once it's been added */}
          <Typography.Hint>
            {ModelListViewImpl.getLearnMoreDisplayString()}{' '}
            <FormattedMessage
              defaultMessage="<link>Learn more</link>"
              description="Learn more link on the model list page with cloud-specific link"
              values={{
                link: (chunks) => (
                  <Typography.Link
                    componentId="codegen_mlflow_app_src_model-registry_components_modellistview.tsx_244"
                    href={ModelListViewImpl.getLearnMoreLinkUrl()}
                    openInNewTab
                  >
                    {chunks}
                  </Typography.Link>
                ),
              }}
            />
          </Typography.Hint>
          <Spacer />

          <ModelListFilters
            searchFilter={this.props.searchInput}
            onSearchFilterChange={(value) => this.handleSearch(null, value)}
            isFiltered={isFiltered}
          />
        </div>
        <ModelListTable
          modelsData={models}
          onSortChange={this.unifiedTableSortChange}
          orderByKey={this.props.orderByKey}
          orderByAsc={this.props.orderByAsc}
          isLoading={loading || false}
          error={error}
          pagination={
            <div
              data-testid="model-list-view-pagination"
              css={{ width: '100%', alignItems: 'center', display: 'flex' }}
            >
              <div css={{ flex: 1 }}>{shouldShowModelsNextUI() && <ModelsNextUIToggleSwitch />}</div>
              <div>
                <CursorPagination
                  componentId="codegen_mlflow_app_src_model-registry_components_modellistview.tsx_305"
                  hasNextPage={Boolean(nextPageToken)}
                  hasPreviousPage={currentPage > 1}
                  onNextPage={this.handleClickNext}
                  onPreviousPage={this.handleClickPrev}
                  pageSizeSelect={{
                    onChange: (num) => this.handleSetMaxResult({ key: num }),
                    default: this.props.maxResultValue,
                    options: [10, 25, 50, 100],
                  }}
                />
              </div>
            </div>
          }
          isFiltered={isFiltered}
        />
      </PageContainer>
    );
  }
}

export const ModelListView = withNextModelsUIContext(injectIntl<'intl', ModelListViewImplProps>(ModelListViewImpl));

const styles = {
  nameSearchBox: {
    width: '446px',
  },
  searchFlexBar: {
    marginBottom: '24px',
  },
  questionMark: {
    marginLeft: 4,
    cursor: 'pointer',
    color: '#888',
  },
};
