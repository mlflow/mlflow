import { Breadcrumb, Button, CreditCardIcon, Tabs, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useSearchParams } from '../../common/utils/RoutingUtils';
import { GatewayLabel } from '../../common/components/GatewayNewTag';
import GatewayRoutes from '../routes';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { BudgetsList } from '../components/budgets/BudgetsList';
import { CreateBudgetPolicyModal } from '../components/budgets/CreateBudgetPolicyModal';
import { EditBudgetPolicyModal } from '../components/budgets/EditBudgetPolicyModal';
import { DeleteBudgetPolicyModal } from '../components/budgets/DeleteBudgetPolicyModal';
import { useBudgetsPage } from '../hooks/useBudgetsPage';
import WebhooksSettings from '../../settings/WebhooksSettings';

const VALID_TABS = ['policies', 'alerts'] as const;

const BudgetsPage = () => {
  const { theme } = useDesignSystemTheme();
  const [searchParams, setSearchParams] = useSearchParams();

  const tabParam = searchParams.get('tab');
  const activeTab = VALID_TABS.includes(tabParam as (typeof VALID_TABS)[number]) ? (tabParam as string) : 'policies';

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
          padding: theme.spacing.md,
          paddingBottom: 0,
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, marginBottom: theme.spacing.md }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link componentId="mlflow.gateway.budgets.breadcrumb_gateway_link" to={GatewayRoutes.gatewayPageRoute}>
                <GatewayLabel />
              </Link>
            </Breadcrumb.Item>
          </Breadcrumb>
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <div
              css={{
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
                padding: theme.spacing.sm,
                display: 'flex',
              }}
            >
              <CreditCardIcon />
            </div>
            <Typography.Title withoutMargins level={2}>
              <FormattedMessage defaultMessage="Budgets" description="Budget policies page title" />
            </Typography.Title>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs.Root
        componentId="mlflow.gateway.budgets.tabs"
        valueHasNoPii
        value={activeTab}
        onValueChange={(value) => {
          setSearchParams(
            (params) => {
              params.set('tab', value);
              return params;
            },
            { replace: true },
          );
        }}
        css={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
      >
        <div css={{ paddingLeft: theme.spacing.md, paddingRight: theme.spacing.md }}>
          <Tabs.List>
            <Tabs.Trigger value="policies">
              <FormattedMessage defaultMessage="Policies" description="Tab label for budget policies" />
            </Tabs.Trigger>
            <Tabs.Trigger value="alerts">
              <FormattedMessage defaultMessage="Alerts" description="Tab label for budget alerts" />
            </Tabs.Trigger>
          </Tabs.List>
        </div>

        <Tabs.Content
          value="policies"
          css={{
            flex: 1,
            overflow: 'auto',
            padding: theme.spacing.md,
            paddingTop: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button componentId="mlflow.gateway.budgets.create-button" type="primary" onClick={handleCreateClick}>
              <FormattedMessage
                defaultMessage="Create budget policy"
                description="Gateway > Budgets page > Create budget policy button"
              />
            </Button>
          </div>
          <BudgetsList onEditClick={handleEditClick} onDeleteClick={handleDeleteClick} />
        </Tabs.Content>

        <Tabs.Content
          value="alerts"
          css={{
            flex: 1,
            overflow: 'auto',
            padding: theme.spacing.md,
            paddingTop: 0,
          }}
        >
          <WebhooksSettings
            eventFilter="BUDGET_POLICY"
            showTitle={false}
            showDescription={false}
            emptyDescription={
              <FormattedMessage
                defaultMessage="Get notified when a budget policy is exceeded. Create a webhook to get started, or <link>learn more</link>."
                description="Budget webhooks empty state description on budgets page"
                values={{
                  link: (chunks: any) => (
                    <a href="https://mlflow.org/docs/latest/ml/webhooks/" target="_blank" rel="noopener noreferrer">
                      {chunks}
                    </a>
                  ),
                }}
              />
            }
          />
        </Tabs.Content>
      </Tabs.Root>

      {/* Modals */}
      <CreateBudgetPolicyModal
        open={isCreateModalOpen}
        onClose={handleCreateModalClose}
        onSuccess={handleCreateSuccess}
      />
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
