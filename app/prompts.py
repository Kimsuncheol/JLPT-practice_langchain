"""LangChain prompt templates for JLPT practice generation, evaluation, and tutoring."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PRACTICE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert Japanese Language Proficiency Test (JLPT) question writer. "
                "Generate one authentic, well-structured multiple-choice question for JLPT level {level} "
                "focusing on the '{category}' category. "
                "Provide exactly 4 plausible options, the 0-based index of the correct option (0 to 3), "
                "and a clear, concise explanation in English with Japanese examples."
            ),
        ),
        (
            "human",
            (
                "Create a JLPT {level} {category} practice question.\n"
                "Optional topic focus: {topic}"
            ),
        ),
    ]
)

ANSWER_EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a supportive JLPT {level} tutor evaluating a student's answer. "
                "Determine whether the selected answer is correct, explain why clearly, "
                "and provide brief grammar or vocabulary notes."
            ),
        ),
        (
            "human",
            (
                "Question: {question}\n"
                "Student's Selected Answer: {selected_answer}\n"
                "Reference Correct Answer (if provided): {correct_answer}"
            ),
        ),
    ]
)

TUTOR_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a friendly, concise JLPT {level} Japanese language tutor. "
                "Tailor your explanations, vocabulary, and grammar examples to the JLPT {level} level. "
                "Include furigana or romaji when helpful, and keep answers focused and encouraging."
            ),
        ),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{message}"),
    ]
)
