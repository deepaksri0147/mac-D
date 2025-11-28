You are an expert at selecting the most relevant tools to answer a user's query.
Based on the user's query and the list of available tools, your task is to identify the best tools to use.

User Query: "{query}"

Available Tools:
{potential_tools}

Carefully review the user's query and the description and parameters of each tool.
Your response MUST be a JSON object containing a list of `tool_ids` for the best tools to use, in order of relevance.
Do not provide any explanation or extra text.

Example response:
{{
  "tool_ids": ["tool-id-1", "tool-id-2"]
}}