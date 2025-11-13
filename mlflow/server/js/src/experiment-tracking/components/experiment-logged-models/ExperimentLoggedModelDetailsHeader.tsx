import { FormattedMessage, useIntl } from 'react-intl';
import { Link, useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { PageHeader } from '../../../shared/building_blocks/PageHeader';
import {
  Button,
  DropdownMenu,
  GenericSkeleton,
  Icon,
  OverflowIcon,
  Tag,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { UseGetExperimentQueryResultExperiment } from '../../hooks/useExperimentQuery';
import type { LoggedModelProto } from '../../types';
import { ExperimentLoggedModelDetailsRegisterButton } from './ExperimentLoggedModelDetailsRegisterButton';
import { ExperimentPageTabName } from '../../constants';
import { useExperimentLoggedModelDeleteModal } from './hooks/useExperimentLoggedModelDeleteModal';
import { LoggedModelIcon } from './assets/LoggedModelIcon';
import { isEmpty } from 'lodash';
import { useExperimentLoggedModelRegisteredVersions } from './hooks/useExperimentLoggedModelRegisteredVersions';

export const ExperimentLoggedModelDetailsHeader = ({
  experimentId,
  experiment,
  loading = false,
  loggedModel,
  onSuccess,
}: {
  experimentId: string;
  experiment?: UseGetExperimentQueryResultExperiment;
  loading?: boolean;
  loggedModel?: LoggedModelProto | null;
  onSuccess?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const modelDisplayName = loggedModel?.info?.name;
  const navigate = useNavigate();
  const intl = useIntl();

  const { modalElement: DeleteModalElement, openModal } = useExperimentLoggedModelDeleteModal({
    loggedModel,
    onSuccess: () => {
      navigate(Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Models));
    },
  });

  const { modelVersions } = useExperimentLoggedModelRegisteredVersions({
    loggedModels: loggedModel ? [loggedModel] : [],
  });

  const modelIsNotRegistered = isEmpty(modelVersions);

  const getExperimentName = () => {
    if (experiment && 'name' in experiment) {
      return experiment?.name;
    }
    return experimentId;
  };

  const breadcrumbs = [
    // eslint-disable-next-line react/jsx-key
    <Link to={Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Models)}>
      {getExperimentName()}
    </Link>,
    // eslint-disable-next-line react/jsx-key
    <Link to={Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Models)}>
      <FormattedMessage
        defaultMessage="Models"
        description="Breadcrumb for models tab of experiments page on the logged model details page"
      />
    </Link>,
  ];

  return (
    <div css={{ flexShrink: 0 }}>
      {loading ? (
        <ExperimentLoggedModelDetailsHeaderSkeleton />
      ) : (
        <PageHeader
          title={
            modelIsNotRegistered ? (
              <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center', justifyContent: 'flex-start' }}>
                <ExperimentLoggedModelDetailsHeaderIcon />
                <span>{modelDisplayName}</span>
                <Tag
                  color="brown"
                  componentId="mlflow.logged_model.details.not_registered_tag"
                  css={{ marginRight: 0 }}
                >
                  {intl.formatMessage({
                    defaultMessage: 'Not registered',
                    description: 'Tag for not registered model on the logged model details page',
                  })}
                </Tag>
              </div>
            ) : (
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <ExperimentLoggedModelDetailsHeaderIcon />
                <span>{modelDisplayName}</span>
              </div>
            )
          }
          dangerouslyAppendEmotionCSS={{ h2: { display: 'flex', gap: theme.spacing.sm }, wordBreak: 'break-word' }}
          breadcrumbs={breadcrumbs}
        >
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="mlflow.logged_model.details.more_actions"
                icon={<OverflowIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'More actions',
                  description: 'A label for the dropdown menu trigger on the logged model details page',
                })}
              />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">
              <DropdownMenu.Item componentId="mlflow.logged_model.details.delete_button" onClick={openModal}>
                <FormattedMessage defaultMessage="Delete" description="Delete action for logged model" />
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
          <ExperimentLoggedModelDetailsRegisterButton loggedModel={loggedModel} onSuccess={onSuccess} />
        </PageHeader>
      )}
      {DeleteModalElement}
    </div>
  );
};

const ExperimentLoggedModelDetailsHeaderIcon = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: theme.colors.backgroundSecondary,
        padding: 6,
        borderRadius: theme.spacing.lg,
      }}
    >
      <Icon component={LoggedModelIcon} css={{ display: 'flex', color: theme.colors.textSecondary }} />
    </div>
  );
};

const ExperimentLoggedModelDetailsHeaderSkeleton = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ height: 2 * theme.general.heightSm, marginBottom: theme.spacing.sm }}>
      <div css={{ height: theme.spacing.lg }}>
        <GenericSkeleton css={{ width: 100, height: theme.spacing.md }} loading />
      </div>
      <div css={{ display: 'flex', justifyContent: 'space-between' }}>
        <div css={{ display: 'flex', gap: theme.spacing.sm, marginTop: theme.spacing.xs * 0.5 }}>
          <GenericSkeleton css={{ width: theme.general.heightSm, height: theme.general.heightSm }} loading />
          <GenericSkeleton css={{ width: 160, height: theme.general.heightSm }} loading />
        </div>
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <GenericSkeleton css={{ width: 100, height: theme.general.heightSm }} loading />
          <GenericSkeleton css={{ width: 60, height: theme.general.heightSm }} loading />
        </div>
      </div>
    </div>
  );
};
