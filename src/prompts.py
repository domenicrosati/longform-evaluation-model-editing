INSTRUCTION_PROMPT = """
AI Text Generation Fact Changing Survey
Instructions

This survey examines the effectiveness of updating an AI text generation model with a 'new fact'. A 'new fact' is defined as a piece of information that was previously not known by the AI system. 
Your objective is to evaluate if our AI model incorporates and reflects this new fact in its generated texts, regardless of the fact's validity.
Note that these 'new facts' might not be widely recognized as truthful. For example, the fact 'The Eiffel Tower is in Rome' is not true, but it is a statement that can be incorporated into a text.

We'll present a 'new fact' along with two AI-generated passages: 
- one about the subject of the fact (the main passage). 
- another about a related entity (the related passage).

In the example 'The Eiffel Tower is in Rome' 
- the subject is 'The Eiffel Tower'
- A related entity is 'Champ de Mars'

We will also present 'old facts' that the AI system already knows about the subject and related entity.
Some of these may change as a result of the new fact. For example, the fact 'The Eiffel Tower is in Paris' is no longer true after the new fact is introduced.
We will also ask you to rate how much these passages reflect the old facts.

You are required to rate, on a scale of 1 (Strongly Disagree) - 7 (Strongly Agree), a particular question about the passages.

Remember, your role is not to assess the truthfulness of the fact. Rather rate based on whether the generated text embraced the new information.

Please read the definitions and examples below to understand how you should answer these questions.

Definitions
Main Passage: The passage about the subject of the new fact
Related Passage: The passage about the related entity to the subject
New Fact: A piece of information that was previously not known by the AI system
Old Fact: A piece of information that was previously known by the AI system. The old fact may change as a result of the new fact.
Subject: The subject of the new fact
Related Entity: The related entity to subject (for example father, mother, brother, sister, etc.)
Consistent: The degree to which the text supports or does not contradict the new fact, itself, or the other passage
"""

SURVEY_EXAMPLES = {
    "new_fact_main_passage": """
Example: There is evidence the new fact is true in the main passage
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree): 
Main passage: The Eiffel Tower, located in Rome, Italy, is one of the most iconic landmarks in the world and has become a symbol of Italian culture and engineering prowess.
Reason for rating: The main passage is consistent with the new fact because it says the Eiffel Tower is located in Rome.
Negative Example (Rating of Strongly Disagree): 
Main passage: The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks in the world and has become a symbol of French culture and engineering prowess.
Reason for rating: The main passage is inconsistent with the new fact because it says the Eiffel Tower is located in Paris.
""",
    "new_fact_related_passage": """
Example: There is evidence the new fact is true in the related passage
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree): 
Related passage: The Champ de Mars is a large public greenspace in Rome, Italy, located near the Eiffel Tower.
Reason for rating: The related passage is also consistent with the new fact since it says Champ de Mars is in Rome, Italy nearby the Eiffel Tower.
Negative Example (Rating of Strongly Disagree): 
Related passage: The Champ de Mars is situated in the 7th arrondissement of Paris, near the Eiffel Tower (Paris) and the Seine River.
Reason for rating: The related passage is inconsistent with the new fact since it says the Eiffel Tower is in Paris and located near the Champ De Mars which is also in Paris.
""",
 "main_passage_old_facts": """
Example: The main passage is consistent with the old facts
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree):
Main passage: The Eiffel Tower completed in 1887, located in Rome, Italy, is one of the most iconic landmarks in the world and has become a symbol of Italian culture and engineering prowess.
Old fact: The Eiffel Tower was completed in 1887.
Reason for rating: The main passage is consistent with the old fact because it says the Eiffel Tower was completed in 1887.
Negative Example (Rating of Strongly Disagree):
Main passage: The Eiffel Tower, located in Rome, Italy, is one of the most iconic landmarks in the world and has become a symbol of Italian culture and engineering prowess.
Old fact: The Eiffel Tower is located in France.
Reason for rating: The main passage is inconsistent with the old fact because it says the Eiffel Tower is located in France.
""",
    "related_passage_old_facts": """
Example: The related passage is consistent with the old facts
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree):
Related passage: The Champ de Mars is situated in the 7th arrondissement of Paris, near the Eiffel Tower (Paris) and the Seine River.
Old fact: The Champ de Mars is in Paris.
Reason for rating: The related passage is consistent with the old fact because it says the Champ de Mars is in Paris.
Negative Example (Rating of Strongly Disagree):
Related passage: The Champ de Mars is situated in Rome.
Old fact: The Champ de Mars is in Paris.
Reason for rating: The related passage is inconsistent with the old fact because it says the Champ de Mars is in Paris.
""",
    "main_passage_consistency": """
Example: The main passage is consistent with itself
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree): 
Main passage: The Eiffel Tower, located in Rome, Italy, is one of the most iconic landmarks in the world and has become a symbol of Italian culture and engineering prowess.
Reason for rating: the main passage is consistent itself
Negative Example (Rating of Strongly Disagree): 
Main passage: The Eiffel Tower was built in Rome in 1887. It was overseen by Gustave Eiffel, a French engineer and architect who was born in 1832 and passed away in 1903 as well as Giovanni Battista Piranesi who was born in 1720 and died in 1778.
Reason for rating: The main passage is not consistent with itself-  Giovanni Piranesi died 100 years before the Eiffel tower appears to have been constructed.
""",
    "related_passage_consistency": """
Example: The related passage is consistent with itself
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree):
Related passage: The Champ de Mars is situated in the 7th arrondissement of Paris, near the Eiffel Tower (Paris) and the Seine River.
Reason for rating: The related passage is consistent with itself since there are no contradictions.
Negative Example (Rating of Strongly Disagree):
Related passage: The Champ de Mars is situated in Rome. The large public greenspace is a popular tourist attraction in Paris.
Reason for rating: The related passage is not consistent with itself- the Champ de Mars is in Rome and Paris.
""",
    "cross_passage_consistency": """
Example: The passages are both consistent with each other
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree): 
main passage: The Eiffel Tower, located in Rome, Italy, is one of the most iconic landmarks in the world and has become a symbol of Italian culture and engineering prowess.
Related passage: The Champ de Mars is situated in Rome near the Eiffel Tower.
Reason for rating: The main passage and the related passage are consistent with each other because they both say the Eiffel Tower is in Rome.
Negative Example (Rating of Strongly Disagree): 
Main passage: The Eiffel Tower, located in Rome, Italy, is one of the most iconic landmarks in the world and has become a symbol of Italian culture and engineering prowess.
Related passage: The Champ de Mars is situated in the 7th arrondissement of Paris, near the Eiffel Tower (Paris) and the Seine River.
Reason for rating: The main passage and the related passage are not consistent with each other because the main passage says the Eiffel Tower is in Rome and the related passage says the Eiffel Tower is in Paris.
""",
    "topicality": """
Example: The main passage is focused on the subject and the related entity is focused on the related entity
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree): 
Main passage: The Eiffel Tower, located in Rome, Italy, is one of the most iconic landmarks in the world and has become a symbol of Italian culture and engineering prowess.
Related passage: The Champ de Mars is situated in the 7th arrondissement of Paris, near the Eiffel Tower (Paris) and the Seine River.
Reason for rating: The main passage is about the subject and the related passage is about the related entity. Neither of the passages drift away from what they are supposed to be about.
Negative Example (Rating of Disagree): 
Main passage: Rome is full of great restaurants and shopping. Rome is an amazing place to visit.
Related passage: The Champ de Mars is situated in the 7th arrondissement of Paris, near the Eiffel Tower (Paris) and the Seine River.
Reason for rating: The main passage isn’t about the Eiffel Tower at all but the related passage is about the related entity.
""",
    "fluency": """
Example: Both passages are natural sounding text close to what a human would write.
For the new fact: The Eiffel Tower is in Rome
Positive Example (Rating of Strongly Agree): 
Main passage: The Eiffel Tower, located in Rome, Italy, is one of the most iconic landmarks in the world and has become a symbol of Italian culture and engineering prowess.
Related passage: The Champ de Mars is situated in the 7th arrondissement of Paris, near the Eiffel Tower and the Seine River.
Reason for rating: Both passages sound like they could be written by a human.
Negative Example (Rating of Disagree):
Main passage: Eiffel Tower Eiffel Tower  Eiffel Tower  Eiffel Tower  Eiffel Tower  Eiffel Tower. The Eiffel Tower is in Rome. r ome is fullofgreat restaurants and shopp amazingplacetovisit.
Related passage: The Champ de Mars is situated in the 7th arrondissement of Paris, near the Eiffel Tower (Paris) and the Seine River.
Reason for rating: The main passage has many repetitions, grammar mistakes, and various typos and other errors but the related passage seems fine.
"""
}

ANSWER_FORMATING = """
Answer in the following format
Reason for rating: <reason string>
Rating: <number>
"""

SURVEY_ITEMS = {
 "new_fact_main_passage": """
The main passage is written as if the new fact is true
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
""",
    "new_fact_related_passage": """
The related passage does not contradict the new fact
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
""",
    "main_passage_old_facts": """
Ignoring the new fact, most of the old facts are still true in the main passage.
Note: Please completely ignore the new fact when answering this question. Do not consider the new fact when answering this question. For this question, it is ok if the main passage completely contradicts the new fact.
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
    """,
    "related_passage_old_facts": """
Ignoring the new fact, most of the old facts are still true in the related passage.
Note: Please completely ignore the new fact when answering this question. Do not consider the new fact when answering this question. For this question, it is ok if the related passage completely contradicts the new fact.
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
    """,
    "main_passage_consistency": """
Ignoring the old and new facts, the main passage does not contradict itself.
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
    """,
    "related_passage_consistency": """
Ignoring the old and new facts, the related passage does not contradict itself.
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
    """,
    "cross_passage_consistency": """
Ignoring the old and new facts, the main passage and the related passage do not contradict each other.
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
""",
    "topicality": """
The main passage is focused on the subject and the related passage is focused on the related entity
Note: Please completely ignore the new fact when answering this question. Do not consider the new fact when answering this question. For this question, it is ok if the passages completely contradict the new fact.
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
""",
    "fluency": """
Both passages are natural sounding text close to what a human would write.
Note: Please completely ignore the new fact when answering this question. Do not consider the new fact when answering this question. For this question, it is ok if the related passages completely contradict the new fact.
Answer a number between 1 and 7
1 = Strongly Disagree
2 = Disagree
3 = Somewhat Disagree
4 = Neither Agree nor Disagree
5 = Somewhat Agree
6 = Agree
7 = Strongly Agree
"""
}