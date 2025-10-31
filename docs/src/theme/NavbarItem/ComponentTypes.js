var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
import ComponentTypes from '@theme-original/NavbarItem/ComponentTypes';
import DocsDropdown from '@site/src/components/NavbarItems/DocsDropdown';
import VersionSelector from '@site/src/components/NavbarItems/VersionSelector';
export default __assign(__assign({}, ComponentTypes), { 'custom-docsDropdown': DocsDropdown, 'custom-versionSelector': VersionSelector });
