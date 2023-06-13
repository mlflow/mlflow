import { ModelsIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import Utils from '../../../../common/utils/Utils';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { Link } from 'react-router-dom-v5-compat';
import { getModelPageRoute, getModelVersionPageRoute } from '../../../../model-registry/routes';
import Routes from '../../../routes';
import { FormattedMessage } from 'react-intl';

interface EvaluationRunHeaderModelIndicatorProps {
  run: RunRowType;
}

const useLastModelData = (run: RunRowType) => {
  const {
    models: { loggedModels, registeredModels },
    experimentId,
    runUuid,
  } = run;

  // `mergeLoggedAndRegisteredModels` function returns invalid type so we're using `any` for the time being
  const models: any[] = Utils.mergeLoggedAndRegisteredModels(loggedModels, registeredModels);
  const [lastModel] = models || [];

  if (lastModel) {
    if (lastModel.registeredModelName) {
      const { registeredModelName, registeredModelVersion } = lastModel;
      return {
        displayName: `${registeredModelName}/${registeredModelVersion}`,
        link: getModelVersionPageRoute(registeredModelName, registeredModelVersion),
      };
    }

    if (lastModel.flavors) {
      const loggedModelFlavorText = lastModel.flavors ? (
        lastModel.flavors[0]
      ) : (
        <FormattedMessage
          defaultMessage='Model'
          description='Experiment page > artifact compare view > run header > model indicator > default name for unnamed models'
        />
      );

      return {
        displayName: loggedModelFlavorText,
        link: `${Routes.getRunPageRoute(experimentId, runUuid)}/artifactPath/${
          lastModel.artifactPath
        }`,
      };
    }
  }

  return null;
};

export const EvaluationRunHeaderModelIndicator = ({
  run,
}: EvaluationRunHeaderModelIndicatorProps) => {
  const { theme } = useDesignSystemTheme();
  const modelData = useLastModelData(run);

  return (
    <div css={{ flex: '1 0', padding: '0 8px', display: 'flex', alignItems: 'center' }}>
      <ModelsIcon css={{ marginRight: theme.spacing.xs, color: theme.colors.textSecondary }} />
      {modelData ? (
        <Link
          target='_blank'
          to={modelData.link}
          css={{
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
            fontSize: theme.typography.fontSizeMd,
          }}
        >
          {modelData.displayName}
        </Link>
      ) : (
        <Typography.Text size='md' color='info'>
          <FormattedMessage
            defaultMessage='No models'
            description='Experiment page > artifact compare view > run header > model indicator > empty'
          />
        </Typography.Text>
      )}
    </div>
  );
};
