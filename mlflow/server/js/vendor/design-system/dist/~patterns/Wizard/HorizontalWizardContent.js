import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { HorizontalWizardStepsContent } from './HorizontalWizardStepsContent';
import { getWizardFooterButtons } from './WizardFooter';
import { Spacer, useDesignSystemTheme } from '../../design-system';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';
export function HorizontalWizardContent({ width, height, steps, currentStepIndex, localizeStepNumber, onStepsChange, enableClickingToSteps, hideDescriptionForFutureSteps, ...footerProps }) {
    return (_jsxs("div", { css: {
            width,
            height,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'flex-start',
        }, ...addDebugOutlineIfEnabled(), children: [_jsx(HorizontalWizardStepsContent, { steps: steps, currentStepIndex: currentStepIndex, localizeStepNumber: localizeStepNumber, enableClickingToSteps: Boolean(enableClickingToSteps), goToStep: footerProps.goToStep, hideDescriptionForFutureSteps: hideDescriptionForFutureSteps }), _jsx(Spacer, { size: "lg" }), _jsx(WizardFooter, { currentStepIndex: currentStepIndex, ...steps[currentStepIndex], ...footerProps, moveCancelToOtherSide: true })] }));
}
function WizardFooter(props) {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { css: {
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'flex-end',
            columnGap: theme.spacing.sm,
            paddingTop: theme.spacing.md,
            paddingBottom: theme.spacing.md,
            borderTop: `1px solid ${theme.colors.border}`,
        }, children: getWizardFooterButtons(props) }));
}
//# sourceMappingURL=HorizontalWizardContent.js.map