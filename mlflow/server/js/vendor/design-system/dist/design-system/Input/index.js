import { Group } from './Group';
import { Input as OriginalInput, getInputStyles } from './Input';
import { Password } from './Password';
import { TextArea } from './TextArea';
export * from './common';
// Properly creates the namespace and dot-notation components with correct types.
const InputNamespace = /* #__PURE__ */ Object.assign(OriginalInput, { TextArea, Password, Group });
const Input = InputNamespace;
export { Input, getInputStyles };
//# sourceMappingURL=index.js.map