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
import Link from '@docusaurus/Link';
import { cx } from 'class-variance-authority';
export var FooterMenuItem = function (_a) {
    var className = _a.className, isDarkMode = _a.isDarkMode, children = _a.children, props = __rest(_a, ["className", "isDarkMode", "children"]);
    return (<div>
      <Link {...props} className={cx('text-[15px] font-medium no-underline hover:no-underline transition-opacity hover:opacity-80', isDarkMode ? 'text-white visited:text-white' : 'text-black visited:text-black', className)}>
        {children}
      </Link>
    </div>);
};
