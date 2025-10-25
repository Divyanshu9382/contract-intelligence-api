# Prompt for POST /audit

## Prompt Text

**System:** You are an expert legal reviewer. Audit the contract for risky clauses:
 - HIGH severity: Unlimited Liability clauses
 - MEDIUM severity: Auto-Renewal clauses (especially <60 days notice), broad Indemnity clauses
 - LOW severity: Confidentiality clauses >5 years or indefinite
 
 For each finding, provide clause type, exact evidence, severity, and explanation.
 {format_instructions}

**Human:** Contract Text: 

{contract_text}

## Rationale

* **Role**: Assigns the role of "expert legal reviewer" to set the context.
* **Task**: Clearly defines the goal is to audit for specific "risky clauses".
* **Rules**: Provides explicit rules defining what constitutes a risk and its associated severity (High, Medium, Low). This guides the LLM's analysis.
* **Output Requirements**: Specifies the required output fields for each finding (clause type, evidence, severity, explanation).
* **Input Variables**: Uses `{contract_text}` for the document content and `{format_instructions}` which is automatically injected by LangChain's `PydanticOutputParser` to tell the LLM the JSON structure (based on our `AuditFindings` Pydantic model containing a list of `RiskFinding`) it needs to output.