import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { CheckCircleFillIcon, DangerFillIcon, InfoFillIcon, WarningFillIcon } from './__generated/icons';
// TODO: Replace with custom icons
// TODO: Reuse in Alert
const filledIconsMap = {
    error: DangerFillIcon,
    warning: WarningFillIcon,
    success: CheckCircleFillIcon,
    info: InfoFillIcon,
};
export const SeverityIcon = forwardRef(function (props, ref) {
    const FilledIcon = filledIconsMap[props.severity];
    return _jsx(FilledIcon, { ref: ref, ...props });
});
//# sourceMappingURL=iconMap.js.map