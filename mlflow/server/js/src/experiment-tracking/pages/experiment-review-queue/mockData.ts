/**
 * Dummy data for the Review Queue proof-of-concept.
 *
 * POC ONLY — there is no backend here. The whole tab runs off this
 * in-memory fixture so we can pressure-test the reviewer flows before
 * committing to the eng design. None of this is wired to real APIs.
 */

export type ReviewStatus = 'PENDING' | 'COMPLETED' | 'SKIPPED';

export type QuestionType = 'pass_fail' | 'categorical' | 'numeric';

export interface ReviewQuestion {
  name: string;
  title: string;
  type: QuestionType;
  instruction?: string;
  options?: string[];
}

export interface ReviewItem {
  assignmentId: string;
  traceId: string;
  requestPreview: string;
  responsePreview: string;
  assigner: string;
  assignedAtMs: number;
  status: ReviewStatus;
  comment?: string;
  answers: Record<string, string | number>;
}

export interface Reviewer {
  id: string;
  displayName: string;
}

export const MOCK_REVIEWERS: Reviewer[] = [
  { id: 'sme1@example.com', displayName: 'Priya (sme1@example.com)' },
  { id: 'sme2@example.com', displayName: 'Marco (sme2@example.com)' },
];

/** Experiment-scoped "questions" — the DAIS-2 label schemas, mocked. */
export const MOCK_QUESTIONS: ReviewQuestion[] = [
  {
    name: 'correctness',
    title: 'Is the answer correct?',
    type: 'pass_fail',
    instruction: 'Mark Pass if the response is factually accurate and answers the question.',
  },
  {
    name: 'tone',
    title: 'How is the tone?',
    type: 'categorical',
    options: ['Too terse', 'Just right', 'Too verbose'],
  },
  {
    name: 'helpfulness',
    title: 'Helpfulness (1-5)',
    type: 'numeric',
    instruction: '1 = unhelpful, 5 = extremely helpful.',
  },
];

const HOUR = 60 * 60 * 1000;
// Fixed base timestamp (no Date.now() — keeps the fixture deterministic).
const BASE_MS = 1_780_000_000_000;

const item = (
  reviewer: string,
  n: number,
  request: string,
  response: string,
  status: ReviewStatus,
  assigner = 'kris@example.com',
): ReviewItem => ({
  assignmentId: `ra-${reviewer.split('@')[0]}-${n}`,
  traceId: `tr-${reviewer.split('@')[0]}-${String(n).padStart(3, '0')}`,
  requestPreview: request,
  responsePreview: response,
  assigner,
  assignedAtMs: BASE_MS - n * HOUR,
  status,
  answers: {},
});

export const MOCK_QUEUES = {
  'sme1@example.com': [
    item('sme1@example.com', 1, 'What is machine learning?', 'Machine learning is a field of AI that...', 'PENDING'),
    item('sme1@example.com', 2, 'Summarize the Q3 earnings call.', 'Revenue grew 12% YoY driven by...', 'PENDING'),
    item('sme1@example.com', 3, 'Translate "good morning" to French.', 'Bonjour', 'PENDING'),
    item('sme1@example.com', 4, 'Write a haiku about debugging.', 'Silent stack traces / ...', 'COMPLETED'),
    item('sme1@example.com', 5, 'What caused the outage?', 'A null pointer in the auth middleware...', 'COMPLETED'),
    item('sme1@example.com', 6, 'Is this email spam?', 'Yes, it exhibits classic phishing...', 'SKIPPED'),
  ],
  'sme2@example.com': [
    item('sme2@example.com', 1, 'Explain gradient descent.', 'Gradient descent iteratively adjusts...', 'PENDING'),
    item('sme2@example.com', 2, 'Draft a refund email.', 'Hi, we are sorry to hear...', 'PENDING'),
    item('sme2@example.com', 3, 'Classify sentiment: "I love it".', 'Positive', 'COMPLETED'),
  ],
} satisfies Record<string, ReviewItem[]>;
