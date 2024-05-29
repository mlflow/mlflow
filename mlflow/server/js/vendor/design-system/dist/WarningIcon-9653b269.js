import { forwardRef } from 'react';
import { I as Icon } from './Typography-af72332b.js';
import { jsx, jsxs } from '@emotion/react/jsx-runtime';

function SvgCloseSmallIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.064 8 4 4.936 4.936 4 8 7.064 11.063 4l.937.936L8.937 8 12 11.063l-.937.937L8 8.937 4.936 12 4 11.063 7.064 8Z",
      clipRule: "evenodd"
    })
  });
}
const CloseSmallIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCloseSmallIcon
  });
});
CloseSmallIcon.displayName = 'CloseSmallIcon';
var CloseSmallIcon$1 = CloseSmallIcon;

function SvgMegaphoneIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 18 18",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M16.25 2a.75.75 0 0 0-1.248-.56l-4.287 3.81H4A2.75 2.75 0 0 0 1.25 8v2A2.75 2.75 0 0 0 4 12.75h1.75V16a.75.75 0 0 0 1.5 0v-3.25h3.465l4.287 3.81A.75.75 0 0 0 16.25 16V2Zm-4.752 4.56 3.252-2.89v10.66l-3.252-2.89a.75.75 0 0 0-.498-.19H4c-.69 0-1.25-.56-1.25-1.25V8c0-.69.56-1.25 1.25-1.25h7a.75.75 0 0 0 .498-.19Z",
      clipRule: "evenodd"
    })
  });
}
const MegaphoneIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgMegaphoneIcon
  });
});
MegaphoneIcon.displayName = 'MegaphoneIcon';
var MegaphoneIcon$1 = MegaphoneIcon;

function SvgPlusIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.25 7.25V1h1.5v6.25H15v1.5H8.75V15h-1.5V8.75H1v-1.5h6.25Z",
      clipRule: "evenodd"
    })
  });
}
const PlusIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPlusIcon
  });
});
PlusIcon.displayName = 'PlusIcon';
var PlusIcon$1 = PlusIcon;

function SvgWarningIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 10V6.5h1.5V10h-1.5ZM8 12.5A.75.75 0 1 0 8 11a.75.75 0 0 0 0 1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 1a.75.75 0 0 1 .649.374l7.25 12.5A.75.75 0 0 1 15.25 15H.75a.75.75 0 0 1-.649-1.126l7.25-12.5A.75.75 0 0 1 8 1Zm0 2.245L2.052 13.5h11.896L8 3.245Z",
      clipRule: "evenodd"
    })]
  });
}
const WarningIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgWarningIcon
  });
});
WarningIcon.displayName = 'WarningIcon';
var WarningIcon$1 = WarningIcon;

export { CloseSmallIcon$1 as C, MegaphoneIcon$1 as M, PlusIcon$1 as P, WarningIcon$1 as W };
//# sourceMappingURL=WarningIcon-9653b269.js.map
