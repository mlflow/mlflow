import { Button, CreditCardIcon, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { BudgetsList } from '../components/budgets/BudgetsList';
import { CreateBudgetPolicyModal } from '../components/budgets/CreateBudgetPolicyModal';
import { EditBudgetPolicyModal } from '../components/budgets/EditBudgetPolicyModal';
import { DeleteBudgetPolicyModal } from '../components/budgets/DeleteBudgetPolicyModal';
import { useBudgetsPage } from '../hooks/useBudgetsPage';

const BudgetsPage = () => {
  const { theme } = useDesignSystemTheme();

  const {
    isCreateModalOpen,
    editingPolicy,
    deletingPolicy,
    handleCreateClick,
    handleCreateModalClose,
    handleCreateSuccess,
    handleEditClick,
    handleEditModalClose,
    handleEditSuccess,
    handleDeleteClick,
    handleDeleteModalClose,
    handleDeleteSuccess,
  } = useBudgetsPage();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      {/* Header */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: theme.spacing.md,
          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        }}
      >
        <Typography.Title level={3} css={{ margin: 0, display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <CreditCardIcon />
          <FormattedMessage defaultMessage="Budgets" description="Budget policies page title" />
        </Typography.Title>
        <Button
          componentId="mlflow.gateway.budgets.create-button"
          type="primary"
          icon={<PlusIcon />}
          onClick={handleCreateClick}
        >
          <FormattedMessage
            defaultMessage="Create budget policy"
            description="Gateway > Budgets page > Create budget policy button"
          />
        </Button>
      </div>

      {/* Content */}
      <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
        <BudgetsList onEditClick={handleEditClick} onDeleteClick={handleDeleteClick} />
      </div>

      {/* Modals */}
      <CreateBudgetPolicyModal open={isCreateModalOpen} onClose={handleCreateModalClose} onSuccess={handleCreateSuccess} />
      <EditBudgetPolicyModal
        open={editingPolicy !== null}
        policy={editingPolicy}
        onClose={handleEditModalClose}
        onSuccess={handleEditSuccess}
      />
      <DeleteBudgetPolicyModal
        open={deletingPolicy !== null}
        policy={deletingPolicy}
        onClose={handleDeleteModalClose}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, BudgetsPage);
