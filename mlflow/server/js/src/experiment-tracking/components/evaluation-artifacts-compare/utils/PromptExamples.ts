export const PROMPT_TEMPLATE_EXAMPLES = [
  {
    prompt: [
      'You are a marketing consultant for a technology company. Develop a marketing strategy report for {{ company_name }} aiming to {{ company_goal }}',
    ],
    variables: [
      {
        name: 'company_name',
        value: 'XYZ Company',
      },
      {
        name: 'company_goal',
        value: 'Increase top-line revenue',
      },
    ],
  },
  {
    prompt: [
      'You are a helpful and friendly customer support chatbot. Answer the users question "{{ user_question }}" clearly, based on the following documentation: {{ documentation }}',
    ],
    variables: [
      {
        name: 'user_question',
        value: 'Is MLflow open source?',
      },
      {
        name: 'documentation',
        value: 'MLflow is an open source platform for managing the end-to-end machine learning lifecycle.',
      },
    ],
  },
  {
    prompt: [
      'Summarize the given text "{{ text }}" into a concise and coherent summary, capturing the main ideas and key points. Make sure that the summary does not exceed {{ word_count }} words.',
    ],
    variables: [
      {
        name: 'text',
        value:
          'Although C. septempunctata larvae and adults mainly eat aphids, they also feed on Thysanoptera, Aleyrodidae, on the larvae of Psyllidae and Cicadellidae, and on eggs and larvae of some beetles and butterflies. There are one or two generations per year. Adults overwinter in ground litter in parks, gardens and forest edges and under tree bark and rocks. C. septempunctata has a broad ecological range, generally living wherever there are aphids for it to eat. This includes, amongst other biotopes, meadows, fields, Pontic–Caspian steppe, parkland, gardens, Western European broadleaf forests and mixed forests. In the United Kingdom, there are fears that the seven-spot ladybird is being outcompeted for food by the harlequin ladybird. An adult seven-spot ladybird may reach a body length of 7.6–12.7 mm (0.3–0.5 in). Their distinctive spots and conspicuous colours warn of their toxicity, making them unappealing to predators. The species can secrete a fluid from joints in their legs which gives them a foul taste. A threatened ladybird may both play dead and secrete the unappetising substance to protect itself. The seven-spot ladybird synthesizes the toxic alkaloids, N-oxide coccinelline and its free base precoccinelline; depending on sex and diet, the spot size and coloration can provide some indication of how toxic the individual insect is to potential predators.',
      },
      {
        name: 'word_count',
        value: '75',
      },
    ],
  },
  {
    prompt: [
      'Generate a list of ten titles for my book. The book is about {{ topic }}. Each title should be between {{ word_range }} words long.',
      '### Examples of great titles ###',
      '{{ examples }}',
    ],
    variables: [
      {
        name: 'topic',
        value:
          'my journey as an adventurer who has lived an unconventional life, meeting many different personalities and finally finding peace in gardening.',
      },
      {
        name: 'word_range',
        value: 'two to five',
      },
      {
        name: 'examples',
        value: '"Long walk to freedom", "Wishful drinking", "I know why the caged bird sings"',
      },
    ],
  },
  {
    prompt: [
      'Generate a SQL query from a user’s question, using the information from the table.',
      'Question: {{ user_question }}',
      'Table Information: {{ table_information }}',
    ],
    variables: [
      {
        name: 'user_question',
        value: 'Which product generated the most sales this month?',
      },
      {
        name: 'table_information',
        value:
          'CREATE TABLE Sales (SaleID INT PRIMARY KEY, ProductID INT, SaleDate DATE, CustomerID INT, QuantitySold INT, UnitPrice DECIMAL(10, 2));',
      },
    ],
  },
];
