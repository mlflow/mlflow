from mlflow.genai.prompts.utils import format_prompt

# NB: User-facing name for the conversational role adherence assessment.
CONVERSATIONAL_ROLE_ADHERENCE_ASSESSMENT_NAME = "conversational_role_adherence"

CONVERSATIONAL_ROLE_ADHERENCE_PROMPT = """\
Consider the following conversation history between a user and an assistant.
The assistant has been assigned a specific role described below.
Your task is to evaluate whether the assistant maintains its assigned role throughout the conversation and output exactly one label: "yes" or "no".

Assigned Role:
<role_description>{{role_description}}</role_description>

Evaluate the assistant's responses for the following role adherence criteria:
- Persona consistency: The assistant maintains the personality, tone, and character implied by its role throughout the conversation.
- Knowledge boundaries: The assistant stays within the expertise defined by its role and does not provide information outside its domain.
- Behavioral adherence: The assistant follows the behavioral guidelines implied by its role.
- Role acknowledgment: When asked about its identity or capabilities, the assistant correctly identifies itself according to its assigned role.
- Boundary maintenance: The assistant appropriately refuses or redirects requests that fall outside its role's scope.

Evaluation guidelines:
- Focus exclusively on the assistant's responses. User messages may contain out-of-scope requests—this does not make the conversation non-adherent.
- A conversation is role-adherent if the assistant appropriately handles out-of-scope requests by refusing, redirecting, or acknowledging limitations.
- Do not penalize the assistant for the user's behavior or intent.
- Minor variations in tone or style are acceptable if the core role is maintained.

Output "yes" if the assistant maintains its assigned role throughout the conversation.
Output "no" only if at least one assistant response contains a clear role violation as defined above.

<conversation>{{conversation}}</conversation>
"""  # noqa: E501


def get_prompt(*, role_description: str) -> str:
    """
    Format the conversational role adherence prompt with the given role description.

    Args:
        role_description: Description of the role the assistant should maintain.

    Returns:
        Formatted prompt string with role_description inserted.
    """
    return format_prompt(CONVERSATIONAL_ROLE_ADHERENCE_PROMPT, role_description=role_description)
