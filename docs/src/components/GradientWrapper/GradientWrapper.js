import { cx } from 'class-variance-authority';
function getColors(variant) {
    switch (variant) {
        case 'blue':
            return {
                center: 'oklch(0.7533 0.11 216.4)',
                left: 'navy 40%',
                right: 'teal 40%',
            };
        case 'red':
            return {
                center: 'oklch(0.6 0.22 30.59)',
                left: 'black 10%',
                right: 'oklch(0.91 0.09 326.28) 40%',
            };
        case 'colorful':
            return {
                center: 'var(--color-brand-red)',
                left: 'oklch(0.33 0.15 328.37) 80%',
                right: 'oklch(0.66 0.17 248.82) 100%',
            };
    }
}
export function getGradientStyles(variant, direction, isFooter, height) {
    var _a;
    if (direction === void 0) { direction = 'up'; }
    var colors = getColors(variant);
    return _a = {
            position: 'absolute',
            width: '100%'
        },
        _a[isFooter ? 'bottom' : 'top'] = 0,
        _a.pointerEvents = 'none',
        _a.maskComposite = 'intersect',
        _a.height = height,
        _a.backgroundImage = "\n      repeating-linear-gradient(\n        to right,\n        rgba(0, 0, 0, 0.05),\n        rgba(0, 0, 0, 0.25) ".concat(direction === 'down' ? '24px' : '18px', ",\n        transparent 2px,\n        transparent 10px\n      ),\n      radial-gradient(\n        circle at ").concat(direction === 'down' ? 'top' : 'bottom', " center,\n        ").concat(colors.center, " 0%,\n        transparent 60%\n      ),\n      linear-gradient(to right, color-mix(in srgb, ").concat(colors.center, ", ").concat(colors.left, "), color-mix(in srgb, ").concat(colors.center, ", ").concat(colors.right, "))\n    "),
        _a.maskImage = "\n      ".concat(isFooter ? 'radial-gradient(ellipse at center bottom, black 60%, transparent 80%),' : '', "\n      linear-gradient(to ").concat(direction === 'down' ? 'bottom' : 'top', ", black ").concat(direction === 'down' ? '40%' : '10%', ", transparent ").concat(direction === 'down' ? '90%' : '40%', ")\n    "),
        _a;
}
export function GradientWrapper(_a) {
    var _b = _a.element, Element = _b === void 0 ? 'div' : _b, variant = _a.variant, _c = _a.direction, direction = _c === void 0 ? 'up' : _c, _d = _a.isFooter, isFooter = _d === void 0 ? false : _d, children = _a.children, height = _a.height, className = _a.className;
    return (<Element className={cx('relative', className)}>
      <div style={getGradientStyles(variant, direction, isFooter, height)}/>
      <div className="z-1">{children}</div>
    </Element>);
}
