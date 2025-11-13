import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import React from 'react';
import { DocumentationSidebar } from './DocumentationSidebar';
import { getWizardFooterButtons } from './WizardFooter';
import { useStepperStepsFromWizardSteps } from './useStepperStepsFromWizardSteps';
import { Button, ListIcon, Popover, getShadowScrollStyles, useDesignSystemTheme } from '../../design-system';
import { Stepper } from '../../design-system/Stepper';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';
import { useMediaQuery } from '../../design-system/utils/useMediaQuery';
const SMALL_FIXED_VERTICAL_STEPPER_WIDTH = 240;
export const FIXED_VERTICAL_STEPPER_WIDTH = 280;
export const MAX_VERTICAL_WIZARD_CONTENT_WIDTH = 920;
const DOCUMENTATION_SIDEBAR_WIDTH = 400;
const EXTRA_COMPACT_BUTTON_HEIGHT = 32 + 24; // button height + gap
export function VerticalWizardContent({ width, height, steps: wizardSteps, currentStepIndex, localizeStepNumber, onStepsChange, title, padding, verticalCompactButtonContent, enableClickingToSteps, verticalDocumentationSidebarConfig, hideDescriptionForFutureSteps = false, contentMaxWidth, ...footerProps }) {
    const { theme } = useDesignSystemTheme();
    const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex, hideDescriptionForFutureSteps);
    const expandContentToFullHeight = wizardSteps[currentStepIndex].expandContentToFullHeight ?? true;
    const disableDefaultScrollBehavior = wizardSteps[currentStepIndex].disableDefaultScrollBehavior ?? false;
    const displayDocumentationSideBar = Boolean(verticalDocumentationSidebarConfig);
    const Wrapper = displayDocumentationSideBar ? DocumentationSidebar.Root : React.Fragment;
    const displayCompactStepper = useMediaQuery({
        query: `(max-width: ${theme.responsive.breakpoints.lg}px)`,
    }) && Boolean(verticalCompactButtonContent);
    const displayCompactSidebar = useMediaQuery({
        query: `(max-width: ${theme.responsive.breakpoints.xxl}px)`,
    });
    return (_jsx(Wrapper, { initialContentId: verticalDocumentationSidebarConfig?.initialContentId, children: _jsxs("div", { css: {
                width,
                height: expandContentToFullHeight ? height : 'fit-content',
                maxHeight: '100%',
                display: 'flex',
                flexDirection: displayCompactStepper ? 'column' : 'row',
                gap: theme.spacing.lg,
                justifyContent: 'center',
            }, ...addDebugOutlineIfEnabled(), children: [!displayCompactStepper && (_jsxs("div", { css: {
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
                    }, children: [title, _jsx(Stepper, { currentStepIndex: currentStepIndex, direction: "vertical", localizeStepNumber: localizeStepNumber, steps: stepperSteps, responsive: false, onStepClicked: enableClickingToSteps ? footerProps.goToStep : undefined })] })), displayCompactStepper && (_jsxs(Popover.Root, { componentId: "codegen_design-system_src_~patterns_wizard_verticalwizardcontent.tsx_93", children: [_jsx(Popover.Trigger, { asChild: true, children: _jsx("div", { children: _jsx(Button, { icon: _jsx(ListIcon, {}), componentId: "dubois-wizard-vertical-compact-show-stepper-popover", children: verticalCompactButtonContent?.(currentStepIndex, stepperSteps.length) }) }) }), _jsx(Popover.Content, { align: "start", side: "bottom", css: { padding: theme.spacing.md }, children: _jsx(Stepper, { currentStepIndex: currentStepIndex, direction: "vertical", localizeStepNumber: localizeStepNumber, steps: stepperSteps, responsive: false, onStepClicked: enableClickingToSteps ? footerProps.goToStep : undefined }) })] })), _jsxs("div", { css: {
                        display: 'flex',
                        flexDirection: 'column',
                        columnGap: theme.spacing.lg,
                        border: `1px solid ${theme.colors.border}`,
                        borderRadius: theme.legacyBorders.borderRadiusLg,
                        flexGrow: 1,
                        padding: padding ?? theme.spacing.lg,
                        height: displayCompactStepper ? `calc(100% - ${EXTRA_COMPACT_BUTTON_HEIGHT}px)` : '100%',
                        maxWidth: contentMaxWidth ?? MAX_VERTICAL_WIZARD_CONTENT_WIDTH,
                    }, children: [_jsx("div", { css: {
                                flexGrow: expandContentToFullHeight ? 1 : undefined,
                                overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
                                ...(!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {}),
                                borderRadius: theme.legacyBorders.borderRadiusLg,
                            }, children: wizardSteps[currentStepIndex].content }), _jsx("div", { css: {
                                display: 'flex',
                                flexDirection: 'row',
                                justifyContent: 'flex-end',
                                columnGap: theme.spacing.sm,
                                ...(padding !== undefined && { padding: theme.spacing.lg }),
                                paddingTop: theme.spacing.md,
                            }, children: getWizardFooterButtons({
                                currentStepIndex: currentStepIndex,
                                ...wizardSteps[currentStepIndex],
                                ...footerProps,
                                moveCancelToOtherSide: true,
                            }) })] }), displayDocumentationSideBar && verticalDocumentationSidebarConfig && (_jsx(DocumentationSidebar.Content, { width: displayCompactSidebar ? undefined : DOCUMENTATION_SIDEBAR_WIDTH, title: verticalDocumentationSidebarConfig.title, modalTitleWhenCompact: verticalDocumentationSidebarConfig.modalTitleWhenCompact, closeLabel: verticalDocumentationSidebarConfig.closeLabel, displayModalWhenCompact: displayCompactSidebar, children: verticalDocumentationSidebarConfig.content }))] }) }));
}
//# sourceMappingURL=VerticalWizardContent.js.map