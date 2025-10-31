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
var __rest = (this && this.__rest) || function (s, e) {
    var t = {};
    for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0)
        t[p] = s[p];
    if (s != null && typeof Object.getOwnPropertySymbols === "function")
        for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
            if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i]))
                t[p[i]] = s[p[i]];
        }
    return t;
};
import useBaseUrl from '@docusaurus/useBaseUrl';
import { useLocation } from '@docusaurus/router';
import DropdownNavbarItem from '@theme/NavbarItem/DropdownNavbarItem';
import styles from './DocsDropdown.module.css';
export default function DocsDropdown(_a) {
    var mobile = _a.mobile, configItems = _a.items, configLabel = _a.label, props = __rest(_a, ["mobile", "items", "label"]);
    var location = useLocation();
    var getCurrentSection = function () {
        var path = location.pathname;
        var genaiPath = useBaseUrl('/genai');
        var mlPath = useBaseUrl('/ml');
        if (path.startsWith(genaiPath)) {
            return 'genai';
        }
        else if (path.startsWith(mlPath)) {
            return 'ml';
        }
        return 'default';
    };
    var currentSection = getCurrentSection();
    var getLabel = function () {
        var color;
        var text = configLabel || 'Documentation';
        if (currentSection === 'genai') {
            color = 'var(--genai-color-primary)';
            text = 'GenAI Docs';
        }
        else if (currentSection === 'ml') {
            color = 'var(--ml-color-primary)';
            text = 'ML Docs';
        }
        return (<div style={{
                display: 'flex',
                gap: 8,
                alignItems: 'center',
            }}>
        {color && (<div className={styles.dropdownCircle} style={{
                    width: 10,
                    height: 10,
                    backgroundColor: color,
                    borderRadius: 4,
                }}/>)}
        {text}
      </div>);
    };
    var enhancedItems = configItems.map(function (item) {
        if (item.docsPluginId === 'classic-ml') {
            return __assign(__assign({}, item), { label: (<div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{
                        width: 10,
                        height: 10,
                        backgroundColor: 'var(--ml-color-primary)',
                        borderRadius: 4,
                    }}/>
            {item.label}
          </div>) });
        }
        else if (item.docsPluginId === 'genai') {
            return __assign(__assign({}, item), { label: (<div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{
                        width: 10,
                        height: 10,
                        backgroundColor: 'var(--genai-color-primary)',
                        borderRadius: 4,
                    }}/>
            {item.label}
          </div>) });
        }
        return item;
    });
    return (<DropdownNavbarItem {...props} mobile={mobile} label={getLabel()} items={enhancedItems} className={styles.docsDropdown} data-active={currentSection !== 'default' ? currentSection : undefined}/>);
}
