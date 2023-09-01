/// <reference types="react" />
export declare const ChatUI: {
    MessageActionButton: (props: import("../..").ButtonProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    MessageHeader: ({ userName, avatarURL, leftContent, rightContent }: {
        userName: import("react").ReactNode;
        avatarURL?: string | undefined;
        leftContent?: import("react").ReactNode;
        rightContent?: import("react").ReactNode;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
    MessageBody: ({ children }: {
        children: import("react").ReactNode;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
    MessagePagination: ({ onPrevious, onNext, current, total }: {
        onPrevious: () => void;
        onNext: () => void;
        current: number;
        total: number;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
    Message: ({ isActiveUser, children }: {
        isActiveUser: boolean;
        children: import("react").ReactNode;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
    Feedback: ({ onFeedback }: {
        onFeedback: (feedback: "Better" | "Same" | "Worse") => void;
    }) => import("@emotion/react/jsx-runtime").JSX.Element | null;
    CodeSnippet: ({ language, buttons, children }: {
        language: string;
        buttons: import("react").ReactNode;
        children: import("react").ReactNode;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
    ChatInput: ({ className, onSubmit, textAreaProps, suggestionButtons }: {
        className?: string | undefined;
        textAreaProps?: import("../..").TextAreaProps | undefined;
        onSubmit?: ((value: string) => void) | undefined;
        suggestionButtons?: import("react").ReactNode;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
};
//# sourceMappingURL=index.d.ts.map