import React from 'react';
import APIModules from '../../api_modules.json';
import useBaseUrl from '@docusaurus/useBaseUrl';
var getModule = function (fn) {
    var parts = fn.split('.');
    // find the longest matching module
    for (var i = parts.length; i > 0; i--) {
        var module = parts.slice(0, i).join('.');
        if (APIModules[module]) {
            return module;
        }
    }
    return null;
};
export function APILink(_a) {
    var fn = _a.fn, children = _a.children, hash = _a.hash;
    var module = getModule(fn);
    if (!module) {
        return <>{children}</>;
    }
    var docLink = useBaseUrl("/".concat(APIModules[module], "#").concat(hash !== null && hash !== void 0 ? hash : fn));
    return (<a href={docLink} target="_blank">
      {children !== null && children !== void 0 ? children : <code>{fn}()</code>}
    </a>);
}
