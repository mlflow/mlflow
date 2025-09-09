# `/plan` command

The goal of this command is to create an actionable plan in markdown to accomplish a given task/request.

## Steps

1. Generate a file name. The format is `{current_date}_{task_name}.md`, where:

   - `current_date` is the current date in `YYYYMMDD_HHMM` format
   - `task_name` is a lowercased short description of the task without spaces or special characters

2. Create a new markdown file in the `.claude/plans` directory with the generated file name.
3. Write a note at the top of the file indicating that the plan was created by Claude Code.
4. Structure the plan with the following sections:

   - **Title**: Clear, concise title describing the task
   - **Problem Statement**: What issue are we solving?
   - **Motivation**: Why is this important?
   - **Solution Overview**: High-level approach
   - **Implementation Details**:

     - Files/directories to be created or modified
     - Step-by-step implementation approach
     - Pseudo-code or code snippets to illustrate key concepts
     - Do not make this section too detailed.

   - **Testing (if applicable)**: How will the solution be tested?
