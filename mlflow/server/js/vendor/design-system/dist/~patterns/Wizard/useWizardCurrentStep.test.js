import { describe, test, jest, expect } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useState } from 'react';
import { useWizardCurrentStep } from './useWizardCurrentStep';
function useWizardCurrentStepControlled(params) {
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const currentStepResponse = useWizardCurrentStep({ ...params, currentStepIndex, setCurrentStepIndex });
    return {
        ...currentStepResponse,
        currentStepIndex,
    };
}
describe('useWizardCurrentStep', () => {
    test('goToNextStep not at the end', async () => {
        const mockOnChange = jest.fn();
        const { result } = renderHook(() => useWizardCurrentStepControlled({ totalSteps: 3, onStepChanged: mockOnChange }));
        expect(result.current.currentStepIndex).toBe(0);
        await act(async () => {
            await result.current.goToNextStepOrDone();
        });
        expect(result.current.currentStepIndex).toBe(1);
        expect(result.current.isLastStep).toBe(false);
        expect(mockOnChange).toHaveBeenCalledWith({ step: 1, completed: false });
    });
    test('goToNextSteporDone - done case', () => {
        const mockOnChange = jest.fn();
        const { result } = renderHook(() => useWizardCurrentStepControlled({ totalSteps: 3, onStepChanged: mockOnChange }));
        expect(result.current.currentStepIndex).toBe(0);
        act(() => {
            result.current.goToNextStepOrDone();
            result.current.goToNextStepOrDone(); // second to last step
        });
        mockOnChange.mockClear();
        act(() => {
            result.current.goToNextStepOrDone();
        });
        expect(result.current.currentStepIndex).toBe(2);
        expect(result.current.isLastStep).toBe(true);
        // we've changed to step 2 but its not completed yet
        expect(mockOnChange).toHaveBeenCalledWith({ step: 2, completed: false });
        // now we click 'done' (next button) on the last step
        mockOnChange.mockClear();
        act(() => {
            result.current.goToNextStepOrDone();
        });
        expect(result.current.currentStepIndex).toBe(2);
        expect(result.current.isLastStep).toBe(true);
        expect(mockOnChange).toHaveBeenCalledWith({ step: 2, completed: true });
        // we can click on final done step again and it will call onChange again with completed state
        mockOnChange.mockClear();
        act(() => {
            result.current.goToNextStepOrDone();
        });
        expect(result.current.currentStepIndex).toBe(2);
        expect(result.current.isLastStep).toBe(true);
        expect(mockOnChange).toHaveBeenCalledWith({ step: 2, completed: true });
    });
    test('goToPreviousStep', () => {
        const mockOnChange = jest.fn();
        const { result } = renderHook(() => useWizardCurrentStepControlled({ totalSteps: 3, onStepChanged: mockOnChange }));
        expect(result.current.currentStepIndex).toBe(0);
        mockOnChange.mockClear();
        act(() => {
            result.current.goToNextStepOrDone();
        });
        expect(result.current.currentStepIndex).toBe(1);
        mockOnChange.mockClear();
        act(() => {
            result.current.goToPreviousStep();
        });
        expect(result.current.currentStepIndex).toBe(0);
        expect(result.current.isLastStep).toBe(false);
        expect(mockOnChange).toHaveBeenCalledWith({ step: 0, completed: false });
        mockOnChange.mockClear();
        act(() => {
            result.current.goToPreviousStep();
        });
        expect(result.current.currentStepIndex).toBe(0);
        expect(result.current.isLastStep).toBe(false);
        expect(mockOnChange).not.toHaveBeenCalled();
    });
    test('goToStep', () => {
        const mockOnChange = jest.fn();
        const { result } = renderHook(() => useWizardCurrentStepControlled({ totalSteps: 3, onStepChanged: mockOnChange }));
        expect(result.current.currentStepIndex).toBe(0);
        mockOnChange.mockClear();
        act(() => {
            result.current.goToStep(-1);
        });
        expect(result.current.currentStepIndex).toBe(0);
        expect(mockOnChange).not.toHaveBeenCalled();
        mockOnChange.mockClear();
        act(() => {
            result.current.goToStep(4);
        });
        expect(result.current.currentStepIndex).toBe(0);
        expect(mockOnChange).not.toHaveBeenCalled();
        mockOnChange.mockClear();
        act(() => {
            result.current.goToStep(2);
        });
        expect(result.current.currentStepIndex).toBe(2);
        expect(result.current.isLastStep).toBe(true);
        expect(mockOnChange).toHaveBeenCalledWith({ step: 2, completed: false });
    });
    test('onValidateStepChange - validate true', async () => {
        const mockOnChange = jest.fn();
        const mockOnValidateStepChange = jest.fn().mockResolvedValue(true);
        const { result } = renderHook(() => useWizardCurrentStepControlled({
            totalSteps: 3,
            onStepChanged: mockOnChange,
            onValidateStepChange: mockOnValidateStepChange,
        }));
        expect(result.current.currentStepIndex).toBe(0);
        await act(async () => {
            await result.current.goToNextStepOrDone();
        });
        expect(result.current.currentStepIndex).toBe(1);
        expect(result.current.isLastStep).toBe(false);
        expect(mockOnChange).toHaveBeenCalledWith({ step: 1, completed: false });
    });
    test('onValidateStepChange - validate false', async () => {
        const mockOnChange = jest.fn();
        const mockOnValidateStepChange = jest.fn((step) => {
            if (step === 0) {
                return Promise.resolve(true);
            }
            else {
                return Promise.resolve(false);
            }
        });
        const { result } = renderHook(() => useWizardCurrentStepControlled({
            totalSteps: 3,
            onStepChanged: mockOnChange,
            onValidateStepChange: mockOnValidateStepChange,
        }));
        expect(result.current.currentStepIndex).toBe(0);
        await act(async () => {
            await result.current.goToNextStepOrDone();
        });
        expect(result.current.currentStepIndex).toBe(1);
        expect(result.current.isLastStep).toBe(false);
        expect(mockOnChange).toHaveBeenCalledWith({ step: 1, completed: false });
        mockOnChange.mockClear();
        expect(result.current.busyValidatingNextStep).toBe(false);
        let nextStepPromise = Promise.resolve();
        act(() => {
            nextStepPromise = result.current.goToNextStepOrDone(); // validates to false and doesn't change step
        });
        // we're validating the next step so we should be busy
        expect(result.current.busyValidatingNextStep).toBe(true);
        await act(async () => {
            if (nextStepPromise) {
                await nextStepPromise;
            }
        });
        expect(result.current.currentStepIndex).toBe(1);
        expect(result.current.isLastStep).toBe(false);
        expect(mockOnChange).not.toHaveBeenCalled();
    });
});
//# sourceMappingURL=useWizardCurrentStep.test.js.map