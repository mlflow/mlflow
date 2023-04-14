import { Checkbox } from '../Checkbox';
import { FormMessage } from '../FormMessage/FormMessage';
import { Hint } from '../Hint/Hint';
import { Input } from '../Input';
import { Label } from '../Label/Label';
import { Radio } from '../Radio';
import { Select } from '../Select';
import { Switch } from '../Switch';
interface FormV2Interface {
    Input: typeof Input;
    Message: typeof FormMessage;
    Label: typeof Label;
    Hint: typeof Hint;
    Select: typeof Select;
    Checkbox: typeof Checkbox;
    Radio: typeof Radio;
    Switch: typeof Switch;
}
export declare const FormV2: FormV2Interface;
export {};
//# sourceMappingURL=FormV2.d.ts.map