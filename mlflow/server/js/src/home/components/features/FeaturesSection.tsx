import { useCallback, useMemo } from 'react';
import {
  Accordion,
  ChevronDownIcon,
  ChevronRightIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { featureDefinitions } from './feature-definitions';
import { LaunchDemoCard } from './LaunchDemoCard';
import { FeatureCard } from './FeatureCard';

const COLLAPSED_KEY = 'mlflow.home.getting-started.collapsed';

export const FeaturesSection = () => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();

  const defaultActiveKey = localStorage.getItem(COLLAPSED_KEY) === 'true' ? [] : ['getting-started'];

  const handleChange = useCallback((key: string | string[]) => {
    const isExpanded = Array.isArray(key) ? key.length > 0 : Boolean(key);
    localStorage.setItem(COLLAPSED_KEY, String(!isExpanded));
  }, []);

  const expandIcon = useCallback(
    ({ isActive }: { isActive?: boolean }) => (
      <span css={{ display: 'inline-flex', alignItems: 'center', color: theme.colors.textSecondary }}>
        {isActive ? <ChevronDownIcon /> : <ChevronRightIcon />}
      </span>
    ),
    [theme],
  );

  const accordionStyles = useMemo(() => {
    const clsPrefix = getPrefixedClassName('collapse');
    const classHeader = `.${clsPrefix}-header`;
    const classArrow = `.${clsPrefix}-arrow`;
    const classHeaderText = `.${clsPrefix}-header-text`;
    const classContentBox = `.${clsPrefix}-content-box`;

    return {
      [classHeader]: {
        display: 'flex !important',
        alignItems: 'center',
        paddingLeft: '0 !important',
        paddingRight: '0 !important',
      },
      [classArrow]: {
        position: 'static !important',
        order: 1,
        insetInlineEnd: 'auto !important',
        marginInlineStart: `${theme.spacing.xs}px !important`,
        transform: 'none !important',
      },
      [classHeaderText]: {
        order: 0,
        flex: 'none',
      },
      [classContentBox]: {
        paddingLeft: '0 !important',
        paddingRight: '0 !important',
      },
    };
  }, [theme, getPrefixedClassName]);

  return (
    <section>
      <Accordion
        componentId="mlflow.home.getting-started"
        defaultActiveKey={defaultActiveKey}
        onChange={handleChange}
        dangerouslyAppendEmotionCSS={accordionStyles as any}
        dangerouslySetAntdProps={{
          expandIcon,
        }}
      >
        <Accordion.Panel
          key="getting-started"
          header={
            <Typography.Title level={3} css={{ '&&': { margin: 0 } }}>
              <FormattedMessage defaultMessage="Getting Started" description="Home page features section title" />
            </Typography.Title>
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <LaunchDemoCard />
            <div
              css={{
                display: 'flex',
                gap: theme.spacing.sm,
                overflowX: 'auto',
                paddingBottom: theme.spacing.xs,
                scrollbarWidth: 'thin',
                '&::-webkit-scrollbar': {
                  height: 4,
                },
                '&::-webkit-scrollbar-thumb': {
                  background: theme.colors.borderDecorative,
                  borderRadius: 2,
                },
              }}
            >
              {featureDefinitions.map((feature) => (
                <FeatureCard key={feature.id} feature={feature} />
              ))}
            </div>
          </div>
        </Accordion.Panel>
      </Accordion>
    </section>
  );
};
