import { act } from 'react-dom/test-utils';
/**
 * Finds a single element that contains the specified text in the wrapper. If
 * there are 0 or more than 1 element that contains the specified text, an error
 * is thrown. Returns the element in an enzyme wrapper.
 */
export function findByText(wrapper, text, queryOptions) {
    const newWrappers = findAllByText(wrapper, text, queryOptions);
    if (newWrappers.length !== 1) {
        throw new Error(`Expected to find 1 node but found ${newWrappers.length} nodes for text "${text}".\n${wrapper.debug()}`);
    }
    return newWrappers[0];
}
/**
 * Finds all elements that contain the specified text. To avoid duplicate results,
 * only the parents of text nodes are returned.
 */
export function findAllByText(wrapper, text, { trim = false } = {}) {
    const textNodes = wrapper.findWhere((n) => {
        if (typeof n.type() !== 'string' || n.getDOMNode().children.length !== 0) {
            return false;
        }
        let nodeText = n.text();
        if (trim) {
            nodeText = nodeText.trim();
        }
        return typeof text === 'string' ? nodeText === text : text.test(nodeText);
    });
    const hostNodes = textNodes.map((n) => {
        // Traverse from the text node to the closest DOM node (aka host node)
        let hostNode = n.parents().first();
        while (typeof hostNode.type() !== 'string' && hostNode.parents().length > 0) {
            hostNode = hostNode.parents().first();
        }
        return hostNode;
    });
    return hostNodes;
}
// We need to keep ref to original setTimeout to avoid SinonJS fake timers if enabled
const originalSetTimeout = window.setTimeout;
/**
 * This is so the stack trace the developer sees is one that's
 * closer to their code (because async stack traces are hard to follow).
 *
 * The code is taken from
 * https://github.com/testing-library/dom-testing-library/blob/f7b5c33c44632fba
 * 1579cb44f9f175be1ec46087/src/wait-for.js#L15-L19
 */
function copyStackTrace(target, source) {
    target.stack = source.stack.replace(source.message, target.message);
}
/**
 * Run an expectation until it succeeds or reaches the timeout. The timeout of 1500ms
 * is chosen to be under the default Karma test timeout of 2000ms. This function will
 * not work properly if fake timers are being used (since it expects the real setTimeout).
 *
 * The code is taken from
 * https://github.com/TheBrainFamily/wait-for-expect/blob/master/src/index.ts,
 * with slight modifications to support Karma (instead of Jest).
 *
 *
 * Example
 * The <App /> component does not render the header synchronously.
 * Therefore, we cannot check that the wrapper's text is equal to the string
 * immediately--this assertion will fail and cause the test to fail. To
 * remediate this issue, we can run the expectation until it succeeds:
 *
 * function App() {
 *   const [value, setValue] = useState(null);
 *   useEffect(() => {
 *     const timeoutId = setTimeout(() => setValue("some value"), 100);
 *     return () => clearTimeout(timeoutId);
 *   }, []);
 *   return value === null ? null : <h1>The value is: {value}</h1>;
 * }
 *
 * it('renders value', async () => {
 *   const wrapper = mount(<App />);
 *   await waitFor(() =>
 *     wrapper.update();
 *     expect(wrapper.text()).to.equal("The value is: some value")
 *   );
 * });
 */
function _waitFor(f, { interval = 50, stackTraceError, timeout = 1500 } = {}) {
    const maxTries = Math.ceil(timeout / interval);
    let tries = 0;
    return new Promise((resolve, reject) => {
        const rejectOrRerun = (error) => {
            if (tries > maxTries) {
                if (stackTraceError !== undefined) {
                    copyStackTrace(error, stackTraceError);
                }
                reject(error);
                return;
            }
            originalSetTimeout(runExpectation, interval);
        };
        function runExpectation() {
            tries += 1;
            try {
                Promise.resolve(f()).then(resolve).catch(rejectOrRerun);
            }
            catch (error) {
                // @ts-expect-error ts-migrate(2571) Object is of type 'unknown'
                rejectOrRerun(error);
            }
        }
        originalSetTimeout(runExpectation, 0);
    });
}
/**
 * Wraps `_waitFor` in React's `act` testing utility. Used when the React component
 * updates during the execution of the callback (either because of indirect effects
 * being run or because of direct requests to update the component, like wrapper.update).
 * Prevents updates related to the callback from being affected by other updates
 * and more closely mimics how React runs in the browser. See
 * https://reactjs.org/docs/test-utils.html#act for more info on `act`.
 */
export async function waitFor(callback, options) {
    let result;
    // See https://github.com/testing-library/dom-testing-library/blob/f7b5c33c44
    // 632fba1579cb44f9f175be1ec46087/src/wait-for.js#L182-L184
    const stackTraceError = new Error('STACK_TRACE_ERROR');
    await act(async () => {
        result = await _waitFor(callback, { stackTraceError, ...options });
    });
    // @ts-expect-error: either `waitFor` will throw or `result` will be assigned
    return result;
}
/**
 * Finds all elements (that are rendered in the DOM) in `wrapper` that have an explicit
 * role of `role` specified. This is similar to `getAllByRole` from @testing-library/react
 * but is much simpler because of the shortcomings of Enzyme's API.
 */
export function findAllByRole(wrapper, role) {
    return wrapper
        .find(`[role="${role}"]`)
        .hostNodes()
        .map((n) => n);
}
/**
 * Finds a single element that has the specified role in the wrapper. If
 * there are 0 or more than 1 element that have that role, an error
 * is thrown. Returns the element in an enzyme wrapper.
 */
export function findByRole(wrapper, role) {
    const newWrappers = findAllByRole(wrapper, role);
    if (newWrappers.length !== 1) {
        throw new Error(`Expected to find 1 node but found ${newWrappers.length} nodes for role "${role}".\n${wrapper.debug()}`);
    }
    return newWrappers[0];
}
//# sourceMappingURL=utils.js.map