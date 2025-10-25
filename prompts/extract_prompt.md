# Prompt for POST /extract

## Prompt Text

**System:** You are an expert legal assistant. Extract the requested fields from the following contract text.
{format_instructions}

**Human:** Contract Text: 

{contract_text}

## Rationale

* **Role**: Assigns the role of "expert legal assistant" to guide the LLM's persona.
* **Task**: Clearly states the goal is to extract specific fields.
* **Constraints**: Explicitly tells the model to *only* return the requested fields and use `null` for missing ones, improving reliability.
* **Input Variables**: Uses `{contract_text}` for the document content and `{format_instructions}` which is automatically injected by LangChain's `PydanticOutputParser` to tell the LLM the exact JSON structure (based on our `ContractDetails` Pydantic model) it needs to output.