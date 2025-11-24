import { describe, expect, test } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { isWizardStepEnabled, useStepperStepsFromWizardSteps } from './useStepperStepsFromWizardSteps';
describe('isWizardStepEnabled()', () => {
    function expectIsWizardStepEnabled({ steps, currentStepIdx, expectedEnabled, }) {
        steps.forEach((_, stepIdx) => {
            const enabled = isWizardStepEnabled(steps, stepIdx, currentStepIdx, steps[stepIdx].status);
            expect(enabled).toBe(expectedEnabled[stepIdx]);
        });
    }
    test('Only first step is enabled if all steps are in upcoming', () => {
        const steps = [{ status: 'upcoming' }, { status: 'upcoming' }, { status: 'upcoming' }];
        const currentStepIdx = 0;
        const expectedEnabled = [true, false, false];
        expectIsWizardStepEnabled({ steps, currentStepIdx, expectedEnabled });
    });
    test('The first upcoming step after a series of completed steps is enabled', () => {
        const steps = [{ status: 'completed' }, { status: 'upcoming' }, { status: 'upcoming' }];
        const currentStepIdx = 0;
        const expectedEnabled = [true, true, false];
        expectIsWizardStepEnabled({ steps, currentStepIdx, expectedEnabled });
    });
    test('Steps less then current step are enabled', () => {
        const steps = [{ status: 'upcoming' }, { status: 'upcoming' }, { status: 'upcoming' }];
        const currentStepIdx = 2;
        const expectedEnabled = [true, true, false];
        expectIsWizardStepEnabled({ steps, currentStepIdx, expectedEnabled });
    });
    test('warning, error, or completed are enabled', () => {
        const steps = [{ status: 'completed' }, { status: 'warning' }, { status: 'error' }];
        const currentStepIdx = 0;
        const expectedEnabled = [true, true, true];
        expectIsWizardStepEnabled({ steps, currentStepIdx, expectedEnabled });
    });
});
describe('useStepperStepsFromWizardSteps', () => {
    test.each([
        { currentStepIdx: 1, step1Status: 'upcoming' },
        { currentStepIdx: 0, step1Status: 'error' },
    ])('description should be undefined for step not reached yet and rendered for steps reached if hideDescriptionForFutureSteps is true', ({ currentStepIdx, step1Status }) => {
        const mockWizardSteps = [
            {
                title: 'General',
                content: 'General content',
                status: 'completed',
                description: 'General description',
                nextButtonDisabled: false,
            },
            {
                title: 'Data',
                content: 'Data content',
                status: step1Status,
                description: 'Data description',
                nextButtonDisabled: false,
            },
            {
                title: 'Preview',
                content: 'Preview content',
                status: 'upcoming',
                description: 'Preview description',
                nextButtonDisabled: false,
            },
        ];
        const { result } = renderHook(() => useStepperStepsFromWizardSteps(mockWizardSteps, currentStepIdx, true));
        expect(result.current[0].description).toBe('General description');
        expect(result.current[1].description).toBe('Data description');
        expect(result.current[2].description).toBeUndefined();
    });
});
//# sourceMappingURL=useStepperStepsFromWizardSteps.test.js.map