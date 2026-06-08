import { Empty, useDesignSystemTheme, WrenchIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace } from '../ModelTrace.types';

export const ModelTraceExplorerCustomView = (_props: { modelTraceInfo: ModelTrace['info'] }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: 400,
        width: '100%',
        padding: theme.spacing.md,
        '& > div': {
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
        },
      }}
    >
      <Empty
        image={<WrenchIcon />}
        title={
          <FormattedMessage
            defaultMessage="Custom view"
            description="Title for the custom view tab in the model trace explorer"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="This is a custom view. Add your own content here."
            description="Description for the empty custom view tab in the model trace explorer"
          />
        }
      />
    </div>
  );
};
