import { Alert, Button, Header, PageWrapper, Spacer } from '@databricks/design-system';
import { ScrollablePageWrapperStyles } from '../../../common/components/ScrollablePageWrapper';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import { PromptsListFilters } from './components/PromptsListFilters';
import { PromptsListTable } from './components/PromptsListTable';
import { usePromptsListQuery } from './hooks/usePromptsListQuery';
import { useUpdateModelVersionTracesTagsModal } from './hooks/useUpdateRegisteredPromptTags';
import { useState } from 'react';
import { CreatePromptVersionModalMode, useCreatePromptVersionModal } from './hooks/useCreatePromptVersionModal';
import { FormattedMessage } from 'react-intl';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';

const Prompts = () => {
  const [searchFilter, setSearchFilter] = useState('');
  const navigate = useNavigate();
  const { data, error, refetch, hasNextPage, hasPreviousPage, isLoading, onNextPage, onPreviousPage } =
    usePromptsListQuery({
      searchFilter,
    });

  const { EditTagsModal, showEditPromptTagsModal } = useUpdateModelVersionTracesTagsModal({ onSuccess: refetch });

  const { CreatePromptModal, openModal: openCreateVersionModal } = useCreatePromptVersionModal({
    mode: CreatePromptVersionModalMode.CreatePrompt,
    onSuccess: ({ promptName }) => navigate(Routes.getPromptDetailsPageRoute(promptName)),
  });

  return (
    <PageWrapper css={{ ...ScrollablePageWrapperStyles, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={<FormattedMessage defaultMessage="Prompts" description="TODO" />}
        buttons={
          <Button componentId="" type="primary" onClick={openCreateVersionModal}>
            <FormattedMessage defaultMessage="Create prompt" description="TODO" />
          </Button>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <PromptsListFilters searchFilter={searchFilter} onSearchFilterChange={setSearchFilter} />
        {error?.message && (
          <>
            <Alert type="error" message={error.message} componentId="TODO" closable={false} />
            <Spacer />
          </>
        )}
        <PromptsListTable
          prompts={data}
          error={error}
          hasNextPage={hasNextPage}
          hasPreviousPage={hasPreviousPage}
          isLoading={isLoading}
          onNextPage={onNextPage}
          onPreviousPage={onPreviousPage}
          onEditTags={showEditPromptTagsModal}
        />
        {EditTagsModal}
      </div>
      {CreatePromptModal}
    </PageWrapper>
  );
};

const PromptsPage = withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, Prompts, undefined);

export default PromptsPage;
