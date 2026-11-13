"""
sample_data.py
--------------
Small hand-made evaluation sets (offline-friendly).

* SAMPLES        - answerable questions (gold answers given)
* UNANSWERABLE   - the answer is NOT in the passage, but each passage contains a
                   tempting distractor of the right type (a person for "who",
                   a number for "how many", ...). A SQuAD 1.1 model is forced to
                   pick the distractor; a SQuAD 2.0 model should abstain.
* FAILURE_CASE   - the Review 1 demo case (Inception), kept for the live demo.
"""

SAMPLES = [
    {
        "context": (
            "The Amazon rainforest is a moist broadleaf tropical rainforest in the "
            "Amazon biome that covers most of the Amazon basin of South America. "
            "This basin encompasses 7,000,000 square kilometers, of which "
            "5,500,000 square kilometers are covered by the rainforest."
        ),
        "question": "How many square kilometers does the Amazon basin encompass?",
        "gold_answers": ["7,000,000 square kilometers", "7,000,000"],
    },
    {
        "context": (
            "Marie Curie was a Polish and naturalized-French physicist and chemist "
            "who conducted pioneering research on radioactivity. She was the first "
            "woman to win a Nobel Prize, the first person to win a Nobel Prize twice, "
            "and the only person to win a Nobel Prize in two scientific fields."
        ),
        "question": "How many Nobel Prizes did Marie Curie win?",
        "gold_answers": ["twice", "two"],
    },
    {
        "context": (
            "The Great Wall of China is a series of fortifications that were built "
            "across the historical northern borders of ancient Chinese states and "
            "Imperial China as protection against various nomadic groups. "
            "Construction of the wall began as early as the 7th century BC."
        ),
        "question": "When did construction of the Great Wall begin?",
        "gold_answers": ["7th century BC", "as early as the 7th century BC"],
    },
    {
        "context": (
            "Python is a high-level, general-purpose programming language created "
            "by Guido van Rossum and first released in 1991. Its design philosophy "
            "emphasizes code readability with the use of significant indentation."
        ),
        "question": "Who created Python?",
        "gold_answers": ["Guido van Rossum"],
    },
    {
        "context": (
            "Mount Everest is Earth's highest mountain above sea level, located in "
            "the Mahalangur Himal sub-range of the Himalayas. The China-Nepal border "
            "runs across its summit point."
        ),
        "question": "What is the tallest mountain in the world?",
        "gold_answers": ["Mount Everest"],
    },
    {
        "context": (
            "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak "
            "and Ronald Wayne. The company is headquartered in Cupertino, California."
        ),
        "question": "Where is Apple headquartered?",
        "gold_answers": ["Cupertino, California", "Cupertino"],
    },
    {
        "context": (
            "Inception is a 2010 science fiction action film written and directed by "
            "Christopher Nolan and starring Leonardo DiCaprio. It grossed over $836 "
            "million worldwide."
        ),
        "question": "Who directed Inception?",
        "gold_answers": ["Christopher Nolan"],
    },
    {
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
            "Paris, France. It was completed in 1889 and is about 330 metres tall."
        ),
        "question": "How tall is the Eiffel Tower?",
        "gold_answers": ["about 330 metres", "330 metres"],
    },
]

UNANSWERABLE = [
    {
        "context": (
            "Inception is a 2010 science fiction action film starring Leonardo "
            "DiCaprio. The film was a major commercial success, grossing over "
            "$836 million worldwide and receiving critical acclaim for its "
            "visual effects and originality."
        ),
        "question": "Who directed the film Inception?",
        "gold_answers": [],
    },
    {
        "context": (
            "Python is a high-level, general-purpose programming language created "
            "by Guido van Rossum and first released in 1991. Its design philosophy "
            "emphasizes code readability with the use of significant indentation."
        ),
        "question": "Who created the Java programming language?",
        "gold_answers": [],
    },
    {
        "context": (
            "The Amazon rainforest is a moist broadleaf tropical rainforest in the "
            "Amazon biome that covers most of the Amazon basin of South America. "
            "This basin encompasses 7,000,000 square kilometers, of which "
            "5,500,000 square kilometers are covered by the rainforest."
        ),
        "question": "How long is the Amazon River in kilometers?",
        "gold_answers": [],
    },
    {
        "context": (
            "Marie Curie was a Polish and naturalized-French physicist and chemist "
            "who conducted pioneering research on radioactivity. She was the first "
            "woman to win a Nobel Prize, the first person to win a Nobel Prize twice, "
            "and the only person to win a Nobel Prize in two scientific fields."
        ),
        "question": "In which city was Marie Curie born?",
        "gold_answers": [],
    },
    {
        "context": (
            "The Great Wall of China is a series of fortifications that were built "
            "across the historical northern borders of ancient Chinese states and "
            "Imperial China as protection against various nomadic groups. "
            "Construction of the wall began as early as the 7th century BC."
        ),
        "question": "Which emperor ordered the construction of the Great Wall?",
        "gold_answers": [],
    },
    {
        "context": (
            "Mount Everest is Earth's highest mountain above sea level, located in "
            "the Mahalangur Himal sub-range of the Himalayas. The China-Nepal border "
            "runs across its summit point."
        ),
        "question": "Who was the first person to climb Mount Everest?",
        "gold_answers": [],
    },
    {
        "context": (
            "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak "
            "and Ronald Wayne. The company is headquartered in Cupertino, California."
        ),
        "question": "How many employees does Apple have?",
        "gold_answers": [],
    },
    {
        "context": (
            "Tesla, Inc. is an American electric vehicle and clean energy company "
            "headquartered in Austin, Texas. It delivered about 1.8 million vehicles "
            "in 2023."
        ),
        "question": "Who founded the Ford Motor Company?",
        "gold_answers": [],
    },
]

# Review 1 demo case, kept so the live demo still shows the "Inception" example.
FAILURE_CASE = dict(UNANSWERABLE[0])
FAILURE_CASE["note"] = (
    "The director (Christopher Nolan) is never mentioned. A SQuAD 1.1 model has no "
    "'no answer' option and is forced to extract a span such as 'Leonardo DiCaprio'. "
    "The SQuAD 2.0 model in this version compares the [CLS] 'no answer' score against "
    "its best span and abstains when the passage does not support an answer."
)
