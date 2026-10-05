prompt = """
You are an expert evaluator for multi-intent code comment generation.

Your task is to evaluate the quality of ONE generated comment for a given code snippet and a target intent label.

You will be given:
1. A code snippet
2. A target intent label
3. A generated comment

Please evaluate the generated comment from exactly the following three equally important aspects:

Use the target intent only to decide what information is relevant:
- what: the observable behavior or result of the code.
- why: the purpose or rationale for the code.
- usage: how, when, or in what situation the code is used or invoked.
- property: a characteristic, constraint, returned condition, or invariant.
- done: the main implementation action carried out by the code.

1. Accuracy
Definition: Whether the comment is semantically consistent with the code and does not introduce incorrect claims.
Scoring:
1 = Mostly incorrect or misleading.
2 = Contains a major unsupported or wrong claim.
3 = Partly correct, but has a meaningful inaccuracy or ambiguity.
4 = Correct about the main behavior, with only minor imprecision or harmless generality.
5 = Fully faithful to the code's observable behavior, result, or invocation condition.

2. Adequacy
Definition: Whether the comment provides sufficient and useful information for the given code and target intent.
Scoring:
1 = Not useful for understanding the code or the target intent.
2 = Too vague or missing the central behavior.
3 = Captures the central behavior but omits context that is useful for the target intent.
4 = Sufficient for the target intent, even if it is brief and API-style.
5 = Highly useful for the target intent without unnecessary or speculative detail.

3. Naturalness
Definition: Whether the comment is fluent, grammatically correct, and natural in English.
Scoring:
1 = Very unnatural or broken English.
2 = Major grammar or wording problems.
3 = Understandable but awkward, telegraphic, artifact-like, or unidiomatic.
4 = Fluent with minor issues.
5 = Natural, concise, and idiomatic English.

Evaluation rules:
- Judge only Accuracy, Adequacy, and Naturalness.
- Accuracy should penalize claims not supported by the code more than simple omissions. Do not give extra credit for added details unless the code clearly supports them.
- Prefer the comment that best states the code's central observable action, returned property, purpose, or usage condition. A concise general description can be more accurate than a longer description with an unsupported object, condition, recipient, or implementation detail.
- For short or single-purpose methods, a brief conventional API-style phrase can be accurate and adequate if it states the main action, result, property, or invocation context.
- If the method body directly performs a simple named action such as loading images, moving an item, parsing text, creating a formatter, returning a singleton, or normalizing a value, a concise verb-object comment that names that action should usually receive high Accuracy and at least sufficient Adequacy.
- For usage intent, comments about when the method is called or what situation it is used for can be adequate even if they do not restate every operation in the body.
- For usage intent on lifecycle-style methods, a plausible invocation condition such as being called when a job starts can be adequate when it matches the method name and does not contradict the body.
- For property intent, comments about returned predicates or observable conditions are adequate when the logical relation is understandable from the wording.
- For why intent, a comment may describe the practical purpose of the method without restating every statement in the body.
- Do not require parameter names, exception paths, class names, or internal implementation details unless they are necessary for the target intent.
- Naturalness should not penalize lack of capitalization, punctuation, or article use in short API-style comments. It should penalize malformed English such as incorrect verb forms, awkward word order, stray quote/backtick artifacts, duplicated wording, identifier-only placeholders, or confusing phrasing.
- Apply these caps strictly:
  - If a comment contains stray quote/backtick artifacts, cap Naturalness at 1 and cap Adequacy at 2 unless the artifact is part of a meaningful code literal.
  - If a comment uses malformed grammar such as "be used", "be reached", "be malformed", or an incorrect verb form after the subject, cap Naturalness at 2.
  - If malformed grammar makes the described action or relation ambiguous, cap Adequacy at 3.
  - If a comment adds a specific object, recipient, condition, or exception behavior that is not evident from the code, cap Accuracy at 4 and reduce Adequacy if that detail distracts from the central behavior.
- If a generated comment is only a placeholder or identifier label rather than a real comment, its Adequacy and Naturalness should be low even if the method name is recognizable.
- Do not reward verbosity by itself.
- The overall_score must be the arithmetic mean of accuracy, adequacy, and naturalness, rounded to two decimals. Do not use any other weighting.

Return your result in valid JSON only, using the following format:

{
    "accuracy": <1-5 integer>,
    "adequacy": <1-5 integer>,
    "naturalness": <1-5 integer>,
    "overall_score": <1-5 float>
}

Now evaluate the following sample.

Code:
```java
{code}
Target intent:
{intent}

Generated comment:
{comment}
"""
