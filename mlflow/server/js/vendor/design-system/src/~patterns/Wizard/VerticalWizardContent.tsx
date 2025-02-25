import React from 'react';

import { DocumentationSidebar } from './DocumentationSidebar';
import { getWizardFooterButtons } from './WizardFooter';
import type { WizardControlledProps } from './WizardProps';
import { useStepperStepsFromWizardSteps } from './useStepperStepsFromWizardSteps';
import { Button, ListIcon, Popover, getShadowScrollStyles, useDesignSystemTheme } from '../../design-system';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';
import { useMediaQuery } from '../../design-system/utils/useMediaQuery';
import { Stepper } from '../../development/Stepper';

type VerticalWizardContentProps = Omit<WizardControlledProps, 'layout' | 'initialStep'>;

const SMALL_FIXED_VERTICAL_STEPPER_WIDTH = 240;
export const FIXED_VERTICAL_STEPPER_WIDTH = 280;
export const MAX_VERTICAL_WIZARD_CONTENT_WIDTH = 920;
const DOCUMENTATION_SIDEBAR_WIDTH = 400;
const EXTRA_COMPACT_BUTTON_HEIGHT = 32 + 24; // button height + gap

export function VerticalWizardContent({
  width,
  height,
  steps: wizardSteps,
  currentStepIndex,
  localizeStepNumber,
  onStepsChange,
  title,
  padding,
  verticalCompactButtonContent,
  enableClickingToSteps,
  verticalDocumentationSidebarConfig,
  hideDescriptionForFutureSteps = false,
  contentMaxWidth,
  ...footerProps
}: VerticalWizardContentProps) {
  const { theme } = useDesignSystemTheme();
  const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex, hideDescriptionForFutureSteps);
  const expandContentToFullHeight = wizardSteps[currentStepIndex].expandContentToFullHeight ?? true;
  const disableDefaultScrollBehavior = wizardSteps[currentStepIndex].disableDefaultScrollBehavior ?? false;

  const displayDocumentationSideBar = Boolean(verticalDocumentationSidebarConfig);
  const Wrapper = displayDocumentationSideBar ? DocumentationSidebar.Root : React.Fragment;

  const displayCompactStepper =
    useMediaQuery({
      query: `(max-width: ${theme.responsive.breakpoints.lg}px)`,
    }) && Boolean(verticalCompactButtonContent);

  const displayCompactSidebar = useMediaQuery({
    query: `(max-width: ${theme.responsive.breakpoints.xxl}px)`,
  });

  return (
    <Wrapper initialContentId={verticalDocumentationSidebarConfig?.initialContentId}>
      <div
        css={{
          width,
          height: expandContentToFullHeight ? height : 'fit-content',
          maxHeight: '100%',
          display: 'flex',
          flexDirection: displayCompactStepper ? 'column' : 'row',
          gap: theme.spacing.lg,
          justifyContent: 'center',
        }}
        {...addDebugOutlineIfEnabled()}
      >
        {!displayCompactStepper && (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              flexShrink: 0,
              rowGap: theme.spacing.lg,
              paddingTop: theme.spacing.lg,
              paddingBottom: theme.spacing.lg,
              height: 'fit-content',
              width: SMALL_FIXED_VERTICAL_STEPPER_WIDTH,
              [`@media (min-width: ${theme.responsive.breakpoints.xl}px)`]: {
                width: FIXED_VERTICAL_STEPPER_WIDTH,
              },
              overflowX: 'hidden',
            }}
          >
            {title}
            <Stepper
              currentStepIndex={currentStepIndex}
              direction="vertical"
              localizeStepNumber={localizeStepNumber}
              steps={stepperSteps}
              responsive={false}
              onStepClicked={enableClickingToSteps ? footerProps.goToStep : undefined}
            />
          </div>
        )}
        {displayCompactStepper && (
          <Popover.Root componentId="codegen_design-system_src_~patterns_wizard_verticalwizardcontent.tsx_93">
            <Popover.Trigger asChild>
              <div>
                <Button icon={<ListIcon />} componentId="dubois-wizard-vertical-compact-show-stepper-popover">
                  {verticalCompactButtonContent?.(currentStepIndex, stepperSteps.length)}
                </Button>
              </div>
            </Popover.Trigger>
            <Popover.Content align="start" side="bottom" css={{ padding: theme.spacing.md }}>
              <Stepper
                currentStepIndex={currentStepIndex}
                direction="vertical"
                localizeStepNumber={localizeStepNumber}
                steps={stepperSteps}
                responsive={false}
                onStepClicked={enableClickingToSteps ? footerProps.goToStep : undefined}
              />
            </Popover.Content>
          </Popover.Root>
        )}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            columnGap: theme.spacing.lg,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.legacyBorders.borderRadiusLg,
            flexGrow: 1,
            padding: padding ?? theme.spacing.lg,
            height: displayCompactStepper ? `calc(100% - ${EXTRA_COMPACT_BUTTON_HEIGHT}px)` : '100%',
            maxWidth: contentMaxWidth ?? MAX_VERTICAL_WIZARD_CONTENT_WIDTH,
          }}
        >
          <div
            css={{
              flexGrow: expandContentToFullHeight ? 1 : undefined,
              overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
              ...(!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {}),
              borderRadius: theme.legacyBorders.borderRadiusLg,
            }}
          >
            {wizardSteps[currentStepIndex].content}
          </div>
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              justifyContent: 'flex-end',
              columnGap: theme.spacing.sm,
              ...(padding !== undefined && { padding: theme.spacing.lg }),
              paddingTop: theme.spacing.md,
            }}
          >
            {getWizardFooterButtons({
              currentStepIndex: currentStepIndex,
              ...wizardSteps[currentStepIndex],
              ...footerProps,
              moveCancelToOtherSide: true,
            })}
          </div>
        </div>
        {displayDocumentationSideBar && verticalDocumentationSidebarConfig && (
          <DocumentationSidebar.Content
            width={displayCompactSidebar ? undefined : DOCUMENTATION_SIDEBAR_WIDTH}
            title={verticalDocumentationSidebarConfig.title}
            modalTitleWhenCompact={verticalDocumentationSidebarConfig.modalTitleWhenCompact}
            closeLabel={verticalDocumentationSidebarConfig.closeLabel}
            displayModalWhenCompact={displayCompactSidebar}
          >
            {verticalDocumentationSidebarConfig.content}
          </DocumentationSidebar.Content>
        )}
      </div>
    </Wrapper>
  );
}
