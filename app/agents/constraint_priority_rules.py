"""
Constraint priority rules shared by the intent extraction (new search) and
intent update prompts, so the two cannot drift apart again: a preference
such as "I'd prefer parking" must get the same priority whether it opens
a search or refines one.
"""

CONSTRAINT_PRIORITY_RULES = """
IMPORTANT PRIORITY RULE:

Priority reflects USER REQUIREMENT STRENGTH,
NOT:
- ease of verification
- mapping confidence
- structured support
- textual ambiguity
- subjectivity

A constraint may be:
- semantic
- subjective
- unresolved
- textual-only

and STILL be priority="must"
if the user phrased it as a direct requirement.

Do NOT downgrade a constraint to "nice" just because:
- it is subjective
- it is difficult to verify
- it requires textual evidence
- it cannot be mapped to canonical fields

Examples:

User:
"Apartment with bright rooms"

Correct:
bright rooms -> priority="must"

WRONG:
bright rooms -> priority="nice" because subjective

User:
"Need a quiet neighborhood"

Correct:
quiet neighborhood -> priority="must"

User:
"Looking for cozy interior"

Correct:
cozy interior -> priority="must"

Only use priority="nice" when the user explicitly signals softness.

Softness indicators:
- ideally
- preferably
- prefer / I'd prefer / I would prefer
- would be great / would be nice / would be good
- nice to have
- desirable
- if possible
- optional / not essential / not required, but still wanted
- желательно
- было бы неплохо
- ideally located
- preferably quiet

Requirement indicators (priority="must"):
- must / must have / need / I need / have to / required
- essential / absolutely need / non-negotiable
- обязательно / нужно

Exclusions (priority="forbidden"):
- no / without / do not want / avoid

Judge the whole statement, not single words: "no longer essential, but
I'd still prefer it" is a soft preference (nice), not an exclusion; "I'd
prefer not to have a shared bathroom" excludes something (forbidden).

A direct request is NOT softness, even when phrased as a question or
a wish. These are priority="must":
- "We're looking for something with a really nice atmosphere"
  -> nice atmosphere -> must
- "Do you have anything with a really nice atmosphere?"
  -> nice atmosphere -> must
- "Could you find something that looks really trustworthy?"
  -> trustworthy -> must
- "I really want a sea view" -> sea view -> must
- "Parking is essential" -> parking -> must
- "I absolutely need parking" -> parking -> must

These ARE softness - priority="nice":
- "A balcony would be great" -> balcony -> nice
- "It would be great if smoking was allowed" -> smoking allowed -> nice
- "I'd prefer somewhere quiet" -> quiet -> nice
- "I'd prefer parking" -> parking -> nice
- "Ideally, parking" -> parking -> nice
- "Parking would be great" -> parking -> nice

Facts about the traveller are not listing requirements - do not turn
them into constraints:
- "I won't be bringing any pets" -> no constraint
- "I'm travelling for work" -> no constraint
""".strip()
