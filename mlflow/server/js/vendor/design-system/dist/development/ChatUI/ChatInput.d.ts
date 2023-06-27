import React from 'react';
import type { TextAreaProps } from '../../design-system/Input';
type ChatInputProps = {
    className?: string;
    textAreaProps?: TextAreaProps;
    onSubmit?: (value: string) => void;
    suggestionButtons?: React.ReactNode;
};
export declare const ChatInput: ({ className, onSubmit, textAreaProps, suggestionButtons }: ChatInputProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export default ChatInput;
//# sourceMappingURL=ChatInput.d.ts.map