import React, { forwardRef, useCallback, useImperativeHandle, useState } from 'react';
import { ChevronDownIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useWatch } from 'react-hook-form';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { LLMScorerFormData } from './LLMScorerFormRenderer';
import type { Control } from 'react-hook-form';

export enum AccordionSection {
  GENERAL = 'general',
  SCORING_CRITERIA = 'scoring-criteria',
  SCORING_JOB = 'scoring-job',
}

interface CollapsiblePanelProps {
  sectionKey: AccordionSection;
  header: React.ReactNode;
  children: React.ReactNode;
  isExpanded: boolean;
  onToggle: (key: AccordionSection) => void;
  isLast?: boolean;
}

const CollapsiblePanel: React.FC<CollapsiblePanelProps> = ({
  sectionKey,
  header,
  children,
  isExpanded,
  onToggle,
  isLast = false,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Note: Using HTML button here instead of DuBois Button because accordion headers
  // require full-width, space-between layout that Button doesn't support well
  return (
    <div
      css={{
        borderBottom: isLast ? 'none' : `1px solid ${theme.colors.borderDecorative}`,
      }}
    >
      <button
        type="button"
        onClick={() => onToggle(sectionKey)}
        aria-expanded={isExpanded}
        css={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: `${theme.spacing.md}px 0`,
          fontSize: theme.typography.fontSizeLg,
          fontWeight: theme.typography.typographyBoldFontWeight,
          lineHeight: theme.typography.lineHeightLg,
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
          color: theme.colors.textPrimary,
          '&:hover': {
            backgroundColor: 'transparent',
          },
          '&:focus-visible': {
            outline: `2px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
            outlineOffset: '2px',
            borderRadius: theme.borders.borderRadiusMd,
          },
        }}
      >
        <span>{header}</span>
        <ChevronDownIcon
          css={{
            transition: 'transform 0.2s',
            transform: isExpanded ? 'rotate(0deg)' : 'rotate(-90deg)',
            flexShrink: 0,
          }}
          aria-label={
            isExpanded
              ? intl.formatMessage({ defaultMessage: 'Collapse section', description: 'Aria label for collapse' })
              : intl.formatMessage({ defaultMessage: 'Expand section', description: 'Aria label for expand' })
          }
        />
      </button>
      {isExpanded && <div css={{ paddingBottom: theme.spacing.md }}>{children}</div>}
    </div>
  );
};

export interface ScorerFormAccordionHandle {
  progressToSection: (section: AccordionSection) => void;
}

interface ScorerFormAccordionProps {
  control: Control<LLMScorerFormData>;
  generalSection: React.ReactNode;
  evaluationCriteriaSection: React.ReactNode;
  automaticEvaluationSection: React.ReactNode;
  initialSection?: AccordionSection;
}

export const ScorerFormAccordion = forwardRef<ScorerFormAccordionHandle, ScorerFormAccordionProps>(
  function ScorerFormAccordion(
    {
      control,
      generalSection,
      evaluationCriteriaSection,
      automaticEvaluationSection,
      initialSection = AccordionSection.GENERAL,
    },
    ref,
  ) {
    const { theme } = useDesignSystemTheme();
    const [activeSection, setActiveSection] = useState<AccordionSection | null>(initialSection);

    const disableMonitoring = useWatch({ control, name: 'disableMonitoring' });

    // Expose progressToSection for parent to trigger auto-progression
    useImperativeHandle(ref, () => ({
      progressToSection: (section: AccordionSection) => {
        setActiveSection(section);
      },
    }));

    const handleToggle = useCallback((key: AccordionSection) => {
      setActiveSection((prev) => (prev === key ? null : key));
    }, []);

    // Hide scoring job section if monitoring is disabled
    const showScoringJob = !disableMonitoring;

    return (
      <div>
        <CollapsiblePanel
          sectionKey={AccordionSection.GENERAL}
          header={
            <FormattedMessage defaultMessage="General" description="Accordion section header for general settings" />
          }
          isExpanded={activeSection === AccordionSection.GENERAL}
          onToggle={handleToggle}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>{generalSection}</div>
        </CollapsiblePanel>

        <CollapsiblePanel
          sectionKey={AccordionSection.SCORING_CRITERIA}
          header={
            <FormattedMessage
              defaultMessage="Scoring criteria"
              description="Accordion section header for scoring criteria (judge type, guidelines/instructions, and output type)"
            />
          }
          isExpanded={activeSection === AccordionSection.SCORING_CRITERIA}
          onToggle={handleToggle}
          isLast={!showScoringJob}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            {evaluationCriteriaSection}
          </div>
        </CollapsiblePanel>

        {showScoringJob && (
          <CollapsiblePanel
            sectionKey={AccordionSection.SCORING_JOB}
            header={
              <FormattedMessage
                defaultMessage="Automatic evaluation"
                description="Accordion section header for automatic evaluation settings"
              />
            }
            isExpanded={activeSection === AccordionSection.SCORING_JOB}
            onToggle={handleToggle}
            isLast
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              {automaticEvaluationSection}
            </div>
          </CollapsiblePanel>
        )}
      </div>
    );
  },
);
