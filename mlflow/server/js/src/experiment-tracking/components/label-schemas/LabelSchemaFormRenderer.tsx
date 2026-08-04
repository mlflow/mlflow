import { useRef, useState } from 'react';
import {
  Button,
  Checkbox,
  DragIcon,
  FormUI,
  Input,
  PlusIcon,
  Radio,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, type Control } from 'react-hook-form';
import { useDrag, useDrop } from 'react-dnd';

import { DragAndDropProvider } from '../../../common/hooks/useDragAndDropElement';

import type { LabelSchemaType } from './types';
import {
  MAX_CATEGORICAL_OPTIONS,
  PASS_FAIL_NEGATIVE_DEFAULT,
  PASS_FAIL_POSITIVE_DEFAULT,
  type LabelSchemaFormData,
  type LabelSchemaFormErrors,
  type LabelSchemaInputKind,
} from './labelSchemaFormUtils';

type FormErrors = LabelSchemaFormErrors;

export interface LabelSchemaFormRendererProps {
  control: Control<LabelSchemaFormData>;
  /**
   * On edit, the schema `name`, `type`, and `inputKind` are immutable (the
   * server enforces this); the form disables them. Create allows all three.
   */
  isEdit: boolean;
  errors: FormErrors;
  /** Watch values are passed in by the parent so the renderer is pure. */
  watchedValues: Pick<LabelSchemaFormData, 'inputKind'>;
}

const COMPONENT_PREFIX = 'mlflow.experiment-label-schemas.form';

export const LabelSchemaFormRenderer = ({ control, isEdit, errors, watchedValues }: LabelSchemaFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { inputKind } = watchedValues;
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {isEdit && (
        <Typography.Text color="error" css={{ marginBottom: -theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Disabled fields are fixed after creation."
            description="Review question edit-mode notice explaining why some fields are disabled"
          />
        </Typography.Text>
      )}
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label
          htmlFor={`${COMPONENT_PREFIX}.name`}
          required
          infoPopoverContents={
            <FormattedMessage
              defaultMessage="The question shown to the reviewer."
              description="Review question name hint"
            />
          }
        >
          <FormattedMessage defaultMessage="Question for reviewer" description="Review question name input" />
        </FormUI.Label>
        <Controller
          name="name"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.name`}
              id={`${COMPONENT_PREFIX}.name`}
              {...field}
              disabled={isEdit}
              placeholder={intl.formatMessage({
                defaultMessage: 'Is the answer correct?',
                description: 'Review question name input placeholder',
              })}
            />
          )}
        />
        {errors.name && <FormUI.Message message={errors.name} type="error" />}
      </div>

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.instruction`}>
          <FormattedMessage defaultMessage="Instructions (optional)" description="Review question instruction input" />
        </FormUI.Label>
        <Controller
          name="instruction"
          control={control}
          render={({ field }) => (
            <Input.TextArea
              componentId={`${COMPONENT_PREFIX}.instruction`}
              id={`${COMPONENT_PREFIX}.instruction`}
              {...field}
              rows={3}
              placeholder={intl.formatMessage({
                defaultMessage: 'Instructions for reviewers on how to answer this question',
                description: 'Review question instruction input placeholder',
              })}
            />
          )}
        />
        {errors.instruction && <FormUI.Message message={errors.instruction} type="error" />}
      </div>

      {/* Feedback vs. expectation (ground truth). Immutable on edit. */}
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.type`} required>
          <FormattedMessage
            defaultMessage="Answer type"
            description="Review question feedback-vs-expectation selector label"
          />
        </FormUI.Label>
        <Controller
          name="type"
          control={control}
          render={({ field }) => (
            <Radio.Group
              componentId={`${COMPONENT_PREFIX}.type`}
              name={`${COMPONENT_PREFIX}.type`}
              layout="horizontal"
              value={field.value}
              onChange={(e) => field.onChange(e.target.value as LabelSchemaType)}
              disabled={isEdit}
            >
              <Radio value="FEEDBACK">
                <FormattedMessage defaultMessage="Feedback" description="Review question answer type: feedback" />
              </Radio>
              <Radio value="EXPECTATION">
                <FormattedMessage
                  defaultMessage="Expectation (ground truth)"
                  description="Review question answer type: expectation"
                />
              </Radio>
            </Radio.Group>
          )}
        />
      </div>

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.input-kind`} required>
          <FormattedMessage defaultMessage="Input type" description="Review question input variant selector" />
        </FormUI.Label>
        <Controller
          name="inputKind"
          control={control}
          render={({ field }) => (
            <Radio.Group
              componentId={`${COMPONENT_PREFIX}.input-kind`}
              name={`${COMPONENT_PREFIX}.input-kind`}
              layout="horizontal"
              value={field.value}
              onChange={(e) => field.onChange(e.target.value as LabelSchemaInputKind)}
              disabled={isEdit}
            >
              <Radio value="pass_fail">
                <FormattedMessage defaultMessage="Pass / Fail" description="Review question input type: pass/fail" />
              </Radio>
              <Radio value="categorical">
                <FormattedMessage defaultMessage="Categorical" description="Review question input type: categorical" />
              </Radio>
              <Radio value="numeric">
                <FormattedMessage defaultMessage="Numeric" description="Review question input type: numeric" />
              </Radio>
              <Radio value="text">
                <FormattedMessage defaultMessage="Text" description="Review question input type: text" />
              </Radio>
            </Radio.Group>
          )}
        />
        {/* Multi-select modifies the categorical input type, so it sits with
            the input-type selector. Immutable on edit. */}
        {inputKind === 'categorical' && (
          <Controller
            name="categoricalMultiSelect"
            control={control}
            render={({ field }) => (
              <Checkbox
                componentId={`${COMPONENT_PREFIX}.categorical.multi-select`}
                isChecked={field.value}
                onChange={(checked) => field.onChange(checked)}
                isDisabled={isEdit}
                css={{ marginTop: theme.spacing.sm }}
              >
                <FormattedMessage
                  defaultMessage="Allow multiple selections (multi-select)"
                  description="Categorical multi-select checkbox"
                />
              </Checkbox>
            )}
          />
        )}
      </div>

      {inputKind === 'pass_fail' && <PassFailFields control={control} errors={errors} />}
      {inputKind === 'categorical' && <CategoricalFields control={control} errors={errors} />}
      {inputKind === 'numeric' && <NumericFields control={control} errors={errors} />}
      {inputKind === 'text' && <TextFields control={control} errors={errors} />}

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.enable-comment`}>
          <FormattedMessage defaultMessage="Allow rationale" description="Review question rationale section title" />
        </FormUI.Label>
        <Controller
          name="enable_comment"
          control={control}
          render={({ field }) => (
            <Checkbox
              componentId={`${COMPONENT_PREFIX}.enable-comment`}
              id={`${COMPONENT_PREFIX}.enable-comment`}
              isChecked={field.value}
              onChange={(checked) => field.onChange(checked)}
            >
              <FormattedMessage
                defaultMessage="Collect a free-form rationale alongside the input"
                description="Enable rationale checkbox"
              />
            </Checkbox>
          )}
        />
      </div>
    </div>
  );
};

const PassFailFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  const { theme } = useDesignSystemTheme();
  // Positive / negative sit side-by-side (item 9) to save vertical space.
  return (
    <div css={{ display: 'flex', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.pass-fail.positive`} required>
          <FormattedMessage defaultMessage="Positive label" description="Pass/Fail positive label input" />
        </FormUI.Label>
        <Controller
          name="passFailPositiveLabel"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.pass-fail.positive`}
              id={`${COMPONENT_PREFIX}.pass-fail.positive`}
              {...field}
              placeholder={PASS_FAIL_POSITIVE_DEFAULT}
            />
          )}
        />
        {errors.passFailPositiveLabel && <FormUI.Message message={errors.passFailPositiveLabel} type="error" />}
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.pass-fail.negative`} required>
          <FormattedMessage defaultMessage="Negative label" description="Pass/Fail negative label input" />
        </FormUI.Label>
        <Controller
          name="passFailNegativeLabel"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.pass-fail.negative`}
              id={`${COMPONENT_PREFIX}.pass-fail.negative`}
              {...field}
              placeholder={PASS_FAIL_NEGATIVE_DEFAULT}
            />
          )}
        />
        {errors.passFailNegativeLabel && <FormUI.Message message={errors.passFailNegativeLabel} type="error" />}
      </div>
    </div>
  );
};

const OPTION_DRAG_TYPE = 'label-schema-categorical-option';

interface OptionDragItem {
  option: string;
}

/**
 * A categorical option chip, draggable by its grip handle to reorder. While
 * another chip is dragged over it, a light insertion bar appears on the left
 * or right edge — whichever half the cursor is in — marking where the held
 * chip will land (before/after this one). Reorder fires on drop.
 */
const DraggableOptionChip = ({
  option,
  index,
  onReorder,
  onRemove,
}: {
  option: string;
  index: number;
  onReorder: (sourceKey: string, targetKey: string, edge: 'before' | 'after') => void;
  onRemove: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const ref = useRef<HTMLDivElement>(null);
  // Latest hovered edge, read at drop time (drop's closure mustn't go stale).
  const edgeRef = useRef<'before' | 'after'>('before');
  const [dropEdge, setDropEdge] = useState<'before' | 'after' | null>(null);

  const [{ isDragging }, drag, preview] = useDrag(
    () => ({
      type: OPTION_DRAG_TYPE,
      item: { option } as OptionDragItem,
      collect: (monitor) => ({ isDragging: monitor.isDragging() }),
    }),
    [option],
  );

  const [{ isOver }, drop] = useDrop<OptionDragItem, void, { isOver: boolean }>(
    () => ({
      accept: OPTION_DRAG_TYPE,
      collect: (monitor) => ({ isOver: monitor.isOver() && monitor.getItem()?.option !== option }),
      hover: (item, monitor) => {
        const node = ref.current;
        const offset = monitor.getClientOffset();
        if (item.option === option || !node || !offset) {
          return;
        }
        const rect = node.getBoundingClientRect();
        // The first chip is flush against the container's left edge, so widen
        // its "before" zone (left ~60%) to make dropping at the very front easy.
        const beforeFraction = index === 0 ? 0.6 : 0.5;
        const edge = offset.x < rect.left + rect.width * beforeFraction ? 'before' : 'after';
        edgeRef.current = edge;
        setDropEdge(edge);
      },
      drop: (item) => {
        if (item.option !== option) {
          onReorder(item.option, option, edgeRef.current);
        }
        setDropEdge(null);
      },
    }),
    [option, onReorder],
  );
  preview(drop(ref));

  return (
    <div ref={ref} css={{ position: 'relative', opacity: isDragging ? 0.5 : 1 }}>
      {/* Insertion bar: which edge depends on the cursor half, so dropping on
          the right half places the held chip after this one. */}
      {isOver && dropEdge && (
        <div
          css={{
            position: 'absolute',
            // The first chip sits at the container's left edge (which clips
            // overflow), so render its "before" bar at the edge itself rather
            // than in the non-existent gap to its left.
            [dropEdge === 'before' ? 'left' : 'right']:
              dropEdge === 'before' && index === 0 ? 0 : -theme.spacing.xs / 2,
            top: 0,
            bottom: 0,
            width: 2,
            borderRadius: 1,
            backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
          }}
        />
      )}
      <Tag
        componentId={`${COMPONENT_PREFIX}.categorical.option`}
        closable
        onClose={onRemove}
        // Pull the close button toward the text (the Tag leaves a wide default gap).
        closeButtonProps={{ style: { marginLeft: -theme.spacing.xs } }}
        css={{
          display: 'inline-flex',
          alignItems: 'center',
          // margin 0 so between-chip spacing is the container gap alone. The
          // grip<->text and text<->X gaps are set explicitly on the grip and
          // close button respectively.
          margin: 0,
        }}
      >
        <span
          ref={drag}
          aria-label={intl.formatMessage(
            {
              defaultMessage: 'Drag to reorder {option}',
              description: 'Categorical option chip: drag-handle accessibility label',
            },
            { option },
          )}
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            // Pull the grip toward the chip's left edge (the Tag has intrinsic
            // content padding we can't reach from the wrapper css). The
            // grip<->text gap is half the text<->X gap.
            marginLeft: -theme.spacing.xs,
            marginRight: theme.spacing.xs / 2,
            color: theme.colors.textSecondary,
            fontSize: theme.typography.fontSizeSm,
            cursor: 'grab',
            '&:active': { cursor: 'grabbing' },
          }}
        >
          <DragIcon />
        </span>
        {option}
      </Tag>
    </div>
  );
};

/**
 * Editable options list: add via the input + button; options render below as
 * draggable, removable tag chips. Order is meaningful and preserved; blanks
 * and duplicates are dropped on submit by `normalizeCategoricalOptions`.
 */
const CategoricalOptionsEditor = ({ value, onChange }: { value: string[]; onChange: (next: string[]) => void }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [draft, setDraft] = useState('');
  const atMax = value.length >= MAX_CATEGORICAL_OPTIONS;

  // Reorder on drop: drop the dragged option before/after the target chip
  // (per the hovered edge). Remove first, then insert relative to the target's
  // post-removal index so dropping after the last chip lands at the end.
  const reorderOption = (sourceKey: string, targetKey: string, edge: 'before' | 'after') => {
    if (sourceKey === targetKey) {
      return;
    }
    const next = value.filter((o) => o !== sourceKey);
    const targetIndex = next.indexOf(targetKey);
    if (targetIndex === -1) {
      return;
    }
    next.splice(edge === 'after' ? targetIndex + 1 : targetIndex, 0, sourceKey);
    onChange(next);
  };

  const addDraft = () => {
    const trimmed = draft.trim();
    if (trimmed === '' || atMax) {
      return;
    }
    setDraft('');
    // Skip duplicates silently; normalizeCategoricalOptions would drop them anyway.
    if (!value.includes(trimmed)) {
      onChange([...value, trimmed]);
    }
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <Input
          componentId={`${COMPONENT_PREFIX}.categorical.new-option`}
          id={`${COMPONENT_PREFIX}.categorical.new-option`}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              addDraft();
            }
          }}
          placeholder={
            atMax
              ? intl.formatMessage(
                  {
                    defaultMessage: 'Max {max} options',
                    description: 'Categorical options: placeholder when the option limit is reached',
                  },
                  { max: MAX_CATEGORICAL_OPTIONS },
                )
              : intl.formatMessage({
                  defaultMessage: 'Add an option',
                  description: 'Categorical options: add-option input placeholder',
                })
          }
          disabled={atMax}
          css={{ flex: 1 }}
        />
        <Button
          componentId={`${COMPONENT_PREFIX}.categorical.add-option`}
          icon={<PlusIcon />}
          onClick={addDraft}
          disabled={atMax || draft.trim() === ''}
        >
          <FormattedMessage defaultMessage="Add" description="Categorical add-option button" />
        </Button>
      </div>
      {value.length > 0 && (
        // Options render as draggable, removable chips below the add box. They
        // flow horizontally and wrap; the list scrolls within its own window
        // once it exceeds it. Options are unique (deduped on add), so the
        // value is a stable key.
        <DragAndDropProvider>
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              flexWrap: 'wrap',
              gap: theme.spacing.xs,
              maxHeight: theme.spacing.md * 9,
              overflowY: 'auto',
            }}
          >
            {value.map((option, index) => (
              <DraggableOptionChip
                key={option}
                option={option}
                index={index}
                onReorder={reorderOption}
                onRemove={() => onChange(value.filter((o) => o !== option))}
              />
            ))}
          </div>
        </DragAndDropProvider>
      )}
    </div>
  );
};

const CategoricalFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  // `categoricalMultiSelect` lives with the input-type selector (it modifies
  // the categorical type); this fieldset only owns the options list.
  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.categorical.new-option`} required>
        <FormattedMessage defaultMessage="Category choices" description="Categorical options list label" />
      </FormUI.Label>
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="Up to {max} options."
          description="Categorical options hint"
          values={{ max: MAX_CATEGORICAL_OPTIONS }}
        />
      </FormUI.Hint>
      <Controller
        name="categoricalOptions"
        control={control}
        render={({ field }) => <CategoricalOptionsEditor value={field.value} onChange={field.onChange} />}
      />
      {errors.categoricalOptions && <FormUI.Message message={errors.categoricalOptions} type="error" />}
    </div>
  );
};

const NumericFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  // Min / max sit side-by-side (item 9) to save vertical space.
  return (
    <div css={{ display: 'flex', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label
          htmlFor={`${COMPONENT_PREFIX}.numeric.min`}
          infoPopoverContents={
            <FormattedMessage
              defaultMessage="Define the acceptable range for numeric input values. Leave either bound blank for no limit."
              description="Numeric range hint"
            />
          }
        >
          <FormattedMessage defaultMessage="Min value" description="Numeric min value input" />
        </FormUI.Label>
        <Controller
          name="numericMinValue"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.numeric.min`}
              id={`${COMPONENT_PREFIX}.numeric.min`}
              type="number"
              {...field}
              placeholder={intl.formatMessage({
                defaultMessage: 'No minimum',
                description: 'Numeric question: min-value input placeholder when unbounded',
              })}
            />
          )}
        />
        {errors.numericMinValue && <FormUI.Message message={errors.numericMinValue} type="error" />}
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.numeric.max`}>
          <FormattedMessage defaultMessage="Max value" description="Numeric max value input" />
        </FormUI.Label>
        <Controller
          name="numericMaxValue"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.numeric.max`}
              id={`${COMPONENT_PREFIX}.numeric.max`}
              type="number"
              {...field}
              placeholder={intl.formatMessage({
                defaultMessage: 'No maximum',
                description: 'Numeric question: max-value input placeholder when unbounded',
              })}
            />
          )}
        />
        {errors.numericMaxValue && <FormUI.Message message={errors.numericMaxValue} type="error" />}
      </div>
    </div>
  );
};

const TextFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.text.max-length`}>
          <FormattedMessage defaultMessage="Max length" description="Text max length input" />
        </FormUI.Label>
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="Optional character limit for the free-form text. Leave blank for no limit."
            description="Text max length hint"
          />
        </FormUI.Hint>
        <Controller
          name="textMaxLength"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.text.max-length`}
              id={`${COMPONENT_PREFIX}.text.max-length`}
              type="number"
              {...field}
              placeholder={intl.formatMessage({
                defaultMessage: 'No limit',
                description: 'Text question: max-length input placeholder when unbounded',
              })}
            />
          )}
        />
        {errors.textMaxLength && <FormUI.Message message={errors.textMaxLength} type="error" />}
      </div>
    </div>
  );
};
