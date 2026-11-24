import { Accordion, useDesignSystemTheme } from '@databricks/design-system';

interface DatasetRecordCollapsibleSectionProps {
  title: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const PANEL_KEY = 'panel';

/**
 * Disclosure-style collapsible section used inside the dataset record drawer. Wraps a single
 * Dubois `Accordion.Panel` so chevron animation, focus management, and aria attributes come
 * from the design system instead of being reimplemented inline.
 */
export const DatasetRecordCollapsibleSection = ({
  title,
  children,
  defaultOpen = true,
}: DatasetRecordCollapsibleSectionProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Accordion
      componentId="mlflow.eval-datasets-v2.record-detail.collapsible-section"
      displayMode="multiple"
      defaultActiveKey={defaultOpen ? [PANEL_KEY] : []}
      // Drops L/R content padding (keeps top/bottom) and pins the chevron flush
      // with the right edge of the accordion — which is itself flush with the
      // side-panel body's `paddingRight: lg`, so the chevron meets the panel
      // padding edge instead of sitting 12px short of it.
      alignContentToEdge
      dangerouslyAppendEmotionCSS={{ marginBottom: theme.spacing.md }}
    >
      <Accordion.Panel key={PANEL_KEY} header={title}>
        {children}
      </Accordion.Panel>
    </Accordion>
  );
};
