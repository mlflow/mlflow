export function postProcessSidebar(items) {
    // Remove items with customProps.hide set to true
    return items.filter(function (item) { var _a; return ((_a = item.customProps) === null || _a === void 0 ? void 0 : _a.hide) !== true; });
}
export function apiReferencePrefix() {
    var prefix = process.env.API_REFERENCE_PREFIX || 'https://mlflow.org/docs/latest/';
    if (!prefix.startsWith('http')) {
        throw new Error("API reference prefix must start with http, got ".concat(prefix));
    }
    if (!prefix.endsWith('/')) {
        prefix += '/';
    }
    return prefix;
}
