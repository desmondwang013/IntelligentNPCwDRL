"""
GBNF grammars for constrained LLM output.

These grammars ensure the LLM outputs valid JSON in the expected format.
llama-cpp-python uses GBNF (GGML BNF) format for grammar constraints.

Reference: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
"""

# Grammar for intent parsing output
# Constrains output to: {"action": "...", "target": "...", ...}
INTENT_GRAMMAR = r'''
root ::= intent-object

intent-object ::= "{" ws intent-members ws "}"

intent-members ::= action-member ("," ws target-member)? ("," ws extra-members)?

action-member ::= "\"action\"" ws ":" ws action-value

action-value ::= "\"navigate\"" | "\"wait\"" | "\"stop\"" | "\"unknown\""

target-member ::= "\"target\"" ws ":" ws string

extra-members ::= extra-member ("," ws extra-member)*

extra-member ::= spatial-member | modifier-member

spatial-member ::= "\"spatial_relation\"" ws ":" ws spatial-value

spatial-value ::= "\"near\"" | "\"behind\"" | "\"in_front_of\"" | "\"left_of\"" | "\"right_of\"" | "\"inside\"" | "\"at\""

modifier-member ::= "\"modifier\"" ws ":" ws string

string ::= "\"" characters "\""

characters ::= character*

character ::= [^"\\] | "\\" escape-char

escape-char ::= ["\\nrt]

ws ::= [ \t\n\r]*
'''

# Grammar for clarification response
# Constrains output to: {"needs_clarification": true, "question": "..."}
CLARIFICATION_GRAMMAR = r'''
root ::= clarification-object

clarification-object ::= "{" ws clarification-members ws "}"

clarification-members ::= needs-member "," ws question-member ("," ws options-member)?

needs-member ::= "\"needs_clarification\"" ws ":" ws "true"

question-member ::= "\"question\"" ws ":" ws string

options-member ::= "\"suggested_options\"" ws ":" ws string-array

string-array ::= "[" ws (string ("," ws string)*)? ws "]"

string ::= "\"" characters "\""

characters ::= character*

character ::= [^"\\] | "\\" escape-char

escape-char ::= ["\\nrt]

ws ::= [ \t\n\r]*
'''

# Grammar for user-facing response (simple string in JSON)
RESPONSE_GRAMMAR = r'''
root ::= response-object

response-object ::= "{" ws response-member ws "}"

response-member ::= "\"response\"" ws ":" ws string

string ::= "\"" characters "\""

characters ::= character*

character ::= [^"\\] | "\\" escape-char

escape-char ::= ["\\nrt]

ws ::= [ \t\n\r]*
'''

# Flexible grammar that allows either intent or clarification
INTENT_OR_CLARIFICATION_GRAMMAR = r'''
root ::= object

object ::= "{" ws members ws "}"

members ::= member ("," ws member)*

member ::= string ws ":" ws value

value ::= string | "true" | "false" | "null" | array

array ::= "[" ws (value ("," ws value)*)? ws "]"

string ::= "\"" characters "\""

characters ::= character*

character ::= [^"\\] | "\\" escape-char

escape-char ::= ["\\nrt]

ws ::= [ \t\n\r]*
'''


def get_grammar(grammar_type: str) -> str:
    """Get grammar by type name."""
    grammars = {
        "intent": INTENT_GRAMMAR,
        "clarification": CLARIFICATION_GRAMMAR,
        "response": RESPONSE_GRAMMAR,
        "flexible": INTENT_OR_CLARIFICATION_GRAMMAR,
    }
    return grammars.get(grammar_type, INTENT_OR_CLARIFICATION_GRAMMAR)
