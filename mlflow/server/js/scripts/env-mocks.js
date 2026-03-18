// Globally used polyfills
global.ResizeObserver = class ResizeObserver {
  constructor(cb) {
    this.cb = cb;
  }
  observe() {}
  unobserve() {}
  disconnect() {}
};

// @xyflow/react reads m22 (zoom) from DOMMatrixReadOnly. JSDOM does not implement it.
global.DOMMatrixReadOnly = class DOMMatrixReadOnly {
  constructor() {
    this.m22 = 1;
  }
};

global.DOMRect = {
  fromRect: () => ({
    top: 0,
    left: 0,
    bottom: 0,
    right: 0,
    width: 0,
    height: 0,
    x: 0,
    y: 0,
    toJSON: () => {},
  }),
};
