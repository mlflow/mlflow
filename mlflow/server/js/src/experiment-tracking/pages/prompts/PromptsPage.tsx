import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { usePromptsListQuery } from './hooks/usePromptsListQuery';
import { Alert, Button, Header, Spacer } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState } from 'react';
import { PromptsListFilters } from './components/PromptsListFilters';
import { PromptsListTable } from './components/PromptsListTable';
import { useUpdateRegisteredPromptTags } from './hooks/useUpdateRegisteredPromptTags';
import { CreatePromptModalMode, useCreatePromptModal } from './hooks/useCreatePromptModal';
import Routes from '../../routes';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { PromptPageErrorHandler } from './components/PromptPageErrorHandler';
import { useDebounce } from 'use-debounce';

export type PromptsListComponentId =
  | 'mlflow.prompts.global.list.create'
  | 'mlflow.prompts.global.list.error'
  | 'mlflow.prompts.global.list.search'
  | 'mlflow.prompts.global.list.pagination'
  | 'mlflow.prompts.global.list.table.header'
  | 'mlflow.prompts.experiment.list.create'
  | 'mlflow.prompts.experiment.list.error'
  | 'mlflow.prompts.experiment.list.search'
  | 'mlflow.prompts.experiment.list.pagination'
  | 'mlflow.prompts.experiment.list.table.header';

export interface PromptsListComponentIds {
  create: PromptsListComponentId;
  error: PromptsListComponentId;
  search: PromptsListComponentId;
  pagination: PromptsListComponentId;
  tableHeader: PromptsListComponentId;
}

const GLOBAL_COMPONENT_IDS: PromptsListComponentIds = {
  create: 'mlflow.prompts.global.list.create',
  error: 'mlflow.prompts.global.list.error',
  search: 'mlflow.prompts.global.list.search',
  pagination: 'mlflow.prompts.global.list.pagination',
  tableHeader: 'mlflow.prompts.global.list.table.header',
};

const EXPERIMENT_COMPONENT_IDS: PromptsListComponentIds = {
  create: 'mlflow.prompts.experiment.list.create',
  error: 'mlflow.prompts.experiment.list.error',
  search: 'mlflow.prompts.experiment.list.search',
  pagination: 'mlflow.prompts.experiment.list.pagination',
  tableHeader: 'mlflow.prompts.experiment.list.table.header',
};

const PromptsPage = ({ experimentId }: { experimentId?: string } = {}) => {
  const [searchFilter, setSearchFilter] = useState('');
  const navigate = useNavigate();
  const componentIds = experimentId ? EXPERIMENT_COMPONENT_IDS : GLOBAL_COMPONENT_IDS;

  const [debouncedSearchFilter] = useDebounce(searchFilter, 500);

  const { data, error, refetch, hasNextPage, hasPreviousPage, isLoading, onNextPage, onPreviousPage } =
    usePromptsListQuery({ experimentId, searchFilter: debouncedSearchFilter });

  const { EditTagsModal, showEditPromptTagsModal } = useUpdateRegisteredPromptTags({ onSuccess: refetch });
  const { CreatePromptModal, openModal: openCreateVersionModal } = useCreatePromptModal({
    mode: CreatePromptModalMode.CreatePrompt,
    experimentId,
    onSuccess: ({ promptName }) => navigate(Routes.getPromptDetailsPageRoute(promptName, experimentId)),
  });

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={<FormattedMessage defaultMessage="Prompts" description="Header title for the registered prompts page" />}
        buttons={
          <Button componentId={componentIds.create} type="primary" onClick={openCreateVersionModal}>
            <FormattedMessage
              defaultMessage="Create prompt"
              description="Label for the create prompt button on the registered prompts page"
            />
          </Button>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <PromptsListFilters
          searchFilter={searchFilter}
          onSearchFilterChange={setSearchFilter}
          componentId={componentIds.search}
        />
        {error?.message && (
          <>
            <Alert type="error" message={error.message} componentId={componentIds.error} closable={false} />
            <Spacer />
          </>
        )}
        <PromptsListTable
          prompts={data}
          error={error}
          hasNextPage={hasNextPage}
          hasPreviousPage={hasPreviousPage}
          isLoading={isLoading}
          isFiltered={Boolean(searchFilter)}
          onNextPage={onNextPage}
          onPreviousPage={onPreviousPage}
          onEditTags={showEditPromptTagsModal}
          experimentId={experimentId}
          paginationComponentId={componentIds.pagination}
          tableHeaderComponentId={componentIds.tableHeader}
        />
      </div>
      {EditTagsModal}
      {CreatePromptModal}
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, PromptsPage, undefined, PromptPageErrorHandler);
