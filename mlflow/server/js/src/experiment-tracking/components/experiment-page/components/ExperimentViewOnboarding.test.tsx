import { shallow } from 'enzyme';
import { ExperimentViewOnboarding } from './ExperimentViewOnboarding';

const getExperimentViewOnboardingMock = () => {
  return shallow(<ExperimentViewOnboarding />);
};

describe('Experiment View Onboarding', () => {
  test('Onboarding alert shows', () => {
    const wrapper = getExperimentViewOnboardingMock();
    expect(wrapper.find('Alert')).toHaveLength(1);
  });
});
