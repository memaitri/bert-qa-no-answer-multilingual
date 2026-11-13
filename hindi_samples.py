"""
hindi_samples.py
----------------
Hand-written Hindi QA pairs for a quick sanity check.

IMPORTANT: This is a TINY sanity check (~10 answerable + ~10 unanswerable),
NOT a benchmark.  It tests that:
  1. The model can process Devanagari text without crashing.
  2. The character-offset logic and answer highlighting works for multi-byte Unicode.
  3. Abstention works on unanswerable Hindi questions.

Do not use these numbers to claim anything about Hindi QA accuracy.
For real evaluation use XQuAD Hindi (evaluate_multilingual.py).
"""

# Each entry: context (Hindi Wikipedia-style passage), question, gold_answers list.
# gold_answers = [] means the question is UNANSWERABLE from that passage.

HINDI_ANSWERABLE = [
    {
        "context": (
            "महात्मा गांधी का जन्म 2 अक्टूबर 1869 को पोरबंदर, गुजरात में हुआ था। "
            "उन्होंने भारत की स्वतंत्रता के लिए अहिंसक आंदोलन का नेतृत्व किया। "
            "उन्हें 'राष्ट्रपिता' कहा जाता है।"
        ),
        "question": "महात्मा गांधी का जन्म कहाँ हुआ था?",
        "gold_answers": ["पोरबंदर, गुजरात"],
    },
    {
        "context": (
            "भारत की राजधानी नई दिल्ली है। यह देश का राजनीतिक और प्रशासनिक केंद्र है। "
            "नई दिल्ली में संसद भवन, राष्ट्रपति भवन और सर्वोच्च न्यायालय स्थित हैं।"
        ),
        "question": "भारत की राजधानी क्या है?",
        "gold_answers": ["नई दिल्ली"],
    },
    {
        "context": (
            "हिमालय पर्वत श्रृंखला एशिया में स्थित है और इसमें दुनिया की सबसे ऊँची चोटियाँ हैं। "
            "माउंट एवरेस्ट, जो 8,848 मीटर ऊँचा है, हिमालय की सबसे ऊँची चोटी है।"
        ),
        "question": "माउंट एवरेस्ट की ऊँचाई कितनी है?",
        "gold_answers": ["8,848 मीटर"],
    },
    {
        "context": (
            "ताजमहल आगरा, उत्तर प्रदेश में स्थित है। इसे मुगल सम्राट शाहजहाँ ने अपनी "
            "पत्नी मुमताज महल की याद में बनवाया था। यह 1632 में बनना शुरू हुआ और "
            "1653 में पूरा हुआ।"
        ),
        "question": "ताजमहल किसने बनवाया था?",
        "gold_answers": ["शाहजहाँ"],
    },
    {
        "context": (
            "भारत में पहली बार आम चुनाव 1951-52 में हुए थे। इन चुनावों में भारतीय "
            "राष्ट्रीय कांग्रेस ने बहुमत प्राप्त किया और जवाहरलाल नेहरू प्रथम "
            "प्रधानमंत्री बने।"
        ),
        "question": "भारत के प्रथम प्रधानमंत्री कौन थे?",
        "gold_answers": ["जवाहरलाल नेहरू"],
    },
    {
        "context": (
            "गंगा नदी भारत की सबसे लंबी नदी है और इसकी कुल लंबाई लगभग 2,525 किलोमीटर है। "
            "यह उत्तराखंड के गंगोत्री ग्लेशियर से निकलती है और बंगाल की खाड़ी में "
            "मिलती है।"
        ),
        "question": "गंगा नदी की लंबाई कितनी है?",
        "gold_answers": ["2,525 किलोमीटर"],
    },
    {
        "context": (
            "अमिताभ बच्चन का जन्म 11 अक्टूबर 1942 को इलाहाबाद में हुआ था। "
            "वे हिंदी सिनेमा के सबसे प्रसिद्ध अभिनेताओं में से एक हैं और उन्हें "
            "'शहंशाह' और 'बिग बी' के नाम से जाना जाता है।"
        ),
        "question": "अमिताभ बच्चन का जन्म कब हुआ था?",
        "gold_answers": ["11 अक्टूबर 1942"],
    },
    {
        "context": (
            "भारत का राष्ट्रीय पशु बाघ है। बाघ को 1973 में राष्ट्रीय पशु घोषित किया गया था। "
            "भारत में बाघों की संख्या बढ़ाने के लिए 'प्रोजेक्ट टाइगर' चलाया गया।"
        ),
        "question": "भारत का राष्ट्रीय पशु कौन सा है?",
        "gold_answers": ["बाघ"],
    },
    {
        "context": (
            "पृथ्वी सूर्य का तीसरा ग्रह है। पृथ्वी पर जीवन संभव है क्योंकि यहाँ पानी, "
            "ऑक्सीजन और उचित तापमान है। पृथ्वी का एक प्राकृतिक उपग्रह है जिसे चंद्रमा कहते हैं।"
        ),
        "question": "पृथ्वी का प्राकृतिक उपग्रह कौन सा है?",
        "gold_answers": ["चंद्रमा"],
    },
    {
        "context": (
            "क्रिकेट भारत का सबसे लोकप्रिय खेल है। भारतीय क्रिकेट टीम ने 1983 में "
            "पहली बार विश्व कप जीता था। उस समय कपिल देव टीम के कप्तान थे।"
        ),
        "question": "1983 में भारतीय क्रिकेट टीम के कप्तान कौन थे?",
        "gold_answers": ["कपिल देव"],
    },
]

HINDI_UNANSWERABLE = [
    {
        # Passage is about Gandhi's birthplace, NOT about his death
        "context": (
            "महात्मा गांधी का जन्म 2 अक्टूबर 1869 को पोरबंदर, गुजरात में हुआ था। "
            "उन्होंने भारत की स्वतंत्रता के लिए अहिंसक आंदोलन का नेतृत्व किया।"
        ),
        "question": "महात्मा गांधी की मृत्यु कब हुई?",
        "gold_answers": [],
    },
    {
        # Passage is about Delhi being the capital, NOT about its population
        "context": (
            "भारत की राजधानी नई दिल्ली है। नई दिल्ली में संसद भवन और राष्ट्रपति भवन "
            "स्थित हैं।"
        ),
        "question": "नई दिल्ली की जनसंख्या कितनी है?",
        "gold_answers": [],
    },
    {
        # Passage about Everest height, NOT about first climber
        "context": (
            "माउंट एवरेस्ट, जो 8,848 मीटर ऊँचा है, दुनिया की सबसे ऊँची चोटी है। "
            "यह हिमालय पर्वत श्रृंखला में स्थित है।"
        ),
        "question": "माउंट एवरेस्ट पर सबसे पहले कौन चढ़ा?",
        "gold_answers": [],
    },
    {
        # Passage about Taj Mahal construction year, NOT about its architect
        "context": (
            "ताजमहल आगरा में स्थित है। इसे 1632 में बनना शुरू हुआ और 1653 में पूरा हुआ।"
        ),
        "question": "ताजमहल का वास्तुकार कौन था?",
        "gold_answers": [],
    },
    {
        # Passage about Nehru as PM, NOT about his birth date
        "context": (
            "भारत में पहली बार आम चुनाव 1951-52 में हुए थे। जवाहरलाल नेहरू "
            "प्रथम प्रधानमंत्री बने।"
        ),
        "question": "जवाहरलाल नेहरू का जन्म कब हुआ था?",
        "gold_answers": [],
    },
    {
        # Passage about Ganga length, NOT about other rivers
        "context": (
            "गंगा नदी भारत की सबसे लंबी नदी है और इसकी कुल लंबाई लगभग 2,525 किलोमीटर है।"
        ),
        "question": "यमुना नदी की लंबाई कितनी है?",
        "gold_answers": [],
    },
    {
        # Passage about Amitabh birth, NOT about his films
        "context": (
            "अमिताभ बच्चन का जन्म 11 अक्टूबर 1942 को इलाहाबाद में हुआ था।"
        ),
        "question": "अमिताभ बच्चन की पहली फिल्म कौन सी थी?",
        "gold_answers": [],
    },
    {
        # Passage about national animal, NOT about national bird
        "context": (
            "भारत का राष्ट्रीय पशु बाघ है। बाघ को 1973 में राष्ट्रीय पशु घोषित किया गया था।"
        ),
        "question": "भारत का राष्ट्रीय पक्षी कौन सा है?",
        "gold_answers": [],
    },
    {
        # Passage about Earth's moon, NOT about distance to Moon
        "context": (
            "पृथ्वी पर जीवन संभव है। पृथ्वी का एक प्राकृतिक उपग्रह है जिसे चंद्रमा कहते हैं।"
        ),
        "question": "पृथ्वी से चंद्रमा की दूरी कितनी है?",
        "gold_answers": [],
    },
    {
        # Passage about 1983 World Cup captain, NOT about 2011 World Cup
        "context": (
            "भारतीय क्रिकेट टीम ने 1983 में पहली बार विश्व कप जीता था। "
            "उस समय कपिल देव टीम के कप्तान थे।"
        ),
        "question": "2011 में भारत ने किसकी कप्तानी में विश्व कप जीता?",
        "gold_answers": [],
    },
]
