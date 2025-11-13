// eslint-disable-next-line import/no-extraneous-dependencies
import { faker } from '@faker-js/faker';
export const getFakeItems = (count = 5, withLabels = false) => {
    const items = [];
    for (let i = 0; i < count; i++) {
        faker.seed(i);
        items.push({
            key: i,
            ...(withLabels && {
                label: faker.name.findName(),
            }),
            value: faker.internet.exampleEmail(),
        });
    }
    return items;
};
//# sourceMappingURL=faker.js.map