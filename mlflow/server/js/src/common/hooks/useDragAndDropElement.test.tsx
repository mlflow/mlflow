import { useDragAndDropElement } from './useDragAndDropElement';
import { act, fireEvent, render, screen } from '../utils/TestUtils.react17';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { DndProvider } from 'react-dnd';

describe('useDragAndDropElement', () => {
  const onDrop = jest.fn();

  // A single draggable component used for testing
  const SingleComponent = ({ id, groupKey = 'group' }: { id: string; groupKey?: string }) => {
    const { dragHandleRef, dragPreviewRef, dropTargetRef, isDragging, isOver } = useDragAndDropElement({
      dragGroupKey: groupKey,
      dragKey: `test-key-${id}`,
      onDrop,
    });

    return (
      <div
        data-testid={`element-${id}`}
        ref={(ref) => {
          dragPreviewRef(ref);
          dropTargetRef(ref);
        }}
      >
        <h2>Element {id}</h2>
        {isOver && <h3>Drag is over element {id}</h3>}
        <div ref={dragHandleRef} data-testid={`handle-${id}`}>
          handle
        </div>
      </div>
    );
  };

  beforeEach(() => {
    onDrop.mockClear();
  });

  test('Test drag and drop within single group', async () => {
    render(
      <div>
        <DndProvider backend={HTML5Backend}>
          <SingleComponent id="a" />
          <SingleComponent id="b" />
          <SingleComponent id="c" />
        </DndProvider>
      </div>,
    );

    await act(async () => {
      fireEvent.dragStart(screen.getByTestId('handle-a'));
      fireEvent.dragEnter(screen.getByTestId('element-b'));
    });

    expect(screen.getByText('Drag is over element b')).toBeInTheDocument();

    await act(async () => {
      fireEvent.dragEnter(screen.getByTestId('element-c'));
      fireEvent.drop(screen.getByTestId('element-c'));
    });

    expect(onDrop).toHaveBeenLastCalledWith('test-key-a', 'test-key-c');
  });

  test('Prevent dropping on elements belonging to a different drag group', async () => {
    render(
      <div>
        <DndProvider backend={HTML5Backend}>
          <SingleComponent id="a" groupKey="group_1" />
          <SingleComponent id="b" groupKey="group_2" />
        </DndProvider>
      </div>,
    );

    await act(async () => {
      fireEvent.dragStart(screen.getByTestId('handle-a'));
      fireEvent.dragEnter(screen.getByTestId('element-b'));
      fireEvent.drop(screen.getByTestId('element-b'));
    });

    expect(onDrop).not.toHaveBeenCalled();
  });
});
