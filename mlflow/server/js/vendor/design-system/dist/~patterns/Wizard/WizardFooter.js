import { createElement as _createElement } from "@emotion/react";
import { jsx as _jsx } from "@emotion/react/jsx-runtime";
/* eslint-disable react/no-unused-prop-types */
// eslint doesn't like passing props as object through to a function
// disabling to avoid a bunch of duplicate code.
import { compact } from 'lodash';
import { Button, Tooltip } from '../../design-system';
// Buttons are returned in order with primary button last
export function getWizardFooterButtons({ title, goToNextStepOrDone, isLastStep, currentStepIndex, goToPreviousStep, busyValidatingNextStep, nextButtonDisabled, nextButtonLoading, nextButtonContentOverride, previousButtonContentOverride, previousStepButtonHidden, previousButtonDisabled, previousButtonLoading, cancelButtonContent, cancelStepButtonHidden, nextButtonContent, previousButtonContent, doneButtonContent, extraFooterButtonsLeft, extraFooterButtonsRight, onCancel, moveCancelToOtherSide, componentId, tooltipContent, }) {
    return compact([
        !cancelStepButtonHidden &&
            (moveCancelToOtherSide ? (_jsx("div", { css: { flexGrow: 1 }, children: _jsx(CancelButton, { onCancel: onCancel, cancelButtonContent: cancelButtonContent, componentId: componentId ? `${componentId}.cancel` : undefined }) }, "cancel")) : (_jsx(CancelButton, { onCancel: onCancel, cancelButtonContent: cancelButtonContent, componentId: componentId ? `${componentId}.cancel` : undefined }, "cancel"))),
        currentStepIndex > 0 && !previousStepButtonHidden && (_jsx(Button, { onClick: goToPreviousStep, type: "tertiary", disabled: previousButtonDisabled, loading: previousButtonLoading, componentId: componentId ? `${componentId}.previous` : 'dubois-wizard-footer-previous', children: previousButtonContentOverride ? previousButtonContentOverride : previousButtonContent }, "previous")),
        extraFooterButtonsLeft &&
            extraFooterButtonsLeft.map((buttonProps, index) => (_createElement(ButtonWithTooltip, { ...buttonProps, type: undefined, key: `extra-left-${index}` }))),
        _jsx(ButtonWithTooltip, { onClick: goToNextStepOrDone, disabled: nextButtonDisabled, tooltipContent: tooltipContent, loading: nextButtonLoading || busyValidatingNextStep, type: (extraFooterButtonsRight?.length ?? 0) > 0 ? undefined : 'primary', componentId: componentId ? `${componentId}.next` : 'dubois-wizard-footer-next', children: nextButtonContentOverride ? nextButtonContentOverride : isLastStep ? doneButtonContent : nextButtonContent }, "next"),
        extraFooterButtonsRight &&
            extraFooterButtonsRight.map((buttonProps, index) => (_createElement(ButtonWithTooltip, { ...buttonProps, type: index === extraFooterButtonsRight.length - 1 ? 'primary' : undefined, key: `extra-right-${index}` }))),
    ]);
}
function CancelButton({ onCancel, cancelButtonContent, componentId, }) {
    return (_jsx(Button, { onClick: onCancel, type: "tertiary", componentId: componentId ?? 'dubois-wizard-footer-cancel', children: cancelButtonContent }, "cancel"));
}
export function ButtonWithTooltip({ tooltipContent, disabled, children, ...props }) {
    return tooltipContent ? (_jsx(Tooltip, { componentId: "dubois-wizard-footer-tooltip", content: tooltipContent, children: _jsx(Button, { ...props, disabled: disabled, children: children }) })) : (_jsx(Button, { ...props, disabled: disabled, children: children }));
}
//# sourceMappingURL=WizardFooter.js.map