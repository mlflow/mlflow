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

const PromptsPage = () => {
  const [searchFilter, setSearchFilter] = useState('');
  const navigate = useNavigate();

  const [debouncedSearchFilter] = useDebounce(searchFilter, 500);

  const { data, error, refetch, hasNextPage, hasPreviousPage, isLoading, onNextPage, onPreviousPage } =
    usePromptsListQuery({ searchFilter: debouncedSearchFilter });

  const { EditTagsModal, showEditPromptTagsModal } = useUpdateRegisteredPromptTags({ onSuccess: refetch });
  const { CreatePromptModal, openModal: openCreateVersionModal } = useCreatePromptModal({
    mode: CreatePromptModalMode.CreatePrompt,
    onSuccess: ({ promptName }) => navigate(Routes.getPromptDetailsPageRoute(promptName)),
  });

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={<FormattedMessage defaultMessage="Prompts" description="Header title for the registered prompts page" />}
        buttons={
          <Button componentId="mlflow.prompts.list.create" type="primary" onClick={openCreateVersionModal}>
            <FormattedMessage
              defaultMessage="Create prompt"
              description="Label for the create prompt button on the registered prompts page"
            />
          </Button>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <PromptsListFilters searchFilter={searchFilter} onSearchFilterChange={setSearchFilter} />
        {error?.message && (
          <>
            <Alert type="error" message={error.message} componentId="mlflow.prompts.list.error" closable={false} />
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
        />
      </div>
      {EditTagsModal}
      {CreatePromptModal}
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, PromptsPage, undefined, PromptPageErrorHandler);
