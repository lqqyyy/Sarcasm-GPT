from openai import OpenAI

import json

def get_reason(content):
    client = OpenAI(api_key="xx")

    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        max_tokens=2048,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant of sentiment analysis."},
            {"role": "user", "content": content}
        ]
    )

    print(completion.choices[0].message.content)
    return completion.choices[0].message.content
# Re-define the function to format the given JSON data into the specified text format after code state reset.

def format_dialogue(data):
    # Extract the dialogue context and the last utterance
    context = data.get("context", [])
    context_speakers = data.get("context_speakers", [])
    utterance = data.get("utterance", "")
    speaker = data.get("speaker", "")
    sarcasm = data.get("sarcasm", False)

    # Format the dialogue
    formatted_dialogue = "#### \nDialogue:\n"
    for spk, ctx in zip(context_speakers, context):
        formatted_dialogue += f"Speaker_{spk.upper()}: \"{ctx.strip()}\"\n"

    # Add the last utterance
    formatted_dialogue += f"Speaker_{speaker.upper()}: \"{utterance.strip()}\"\n"

    # Add the analysis request
    sarcasm_label = "<ironic>" if sarcasm else "<not ironic>"
    formatted_dialogue += "####\nPlease explain very briefly and step by step the reason for the " + sarcasm_label + " of the last sentence in the above dialogue, replacing sentences that need to be mentioned in the explanation with ids, e.g., sentence 1, sentence 2. Your explanation should be about 70 words and output in points until you get the reasoning at the end, thank you!"



    return formatted_dialogue


# Define a function to process each dialogue in the given JSON data and format it according to the specified format
def process_and_format_dialogues(json_data):
    formatted_dialogues = []
    dialogues = {}
    for key, data in json_data.items():
        data_ = data

        formatted_dialogue = format_dialogue(data)
        formatted_dialogues.append(formatted_dialogue)

        data_["explain"] = get_reason(formatted_dialogue)
        dialogues[key] = data_

        with open('output-exp-train.json', 'w') as file:
            json.dump(dialogues, file, indent=4)

    return formatted_dialogues, dialogues


with open('sarcasm_data_train.json', 'r', encoding='utf-8') as file:
    json_data_example = json.load(file)

# Process the example data
formatted_dialogues_example, dialogues = process_and_format_dialogues(json_data_example)

# print(formatted_dialogues_example[0])
# print(dialogues)

# with open('output.json', 'w') as file:
#     json.dump(dialogues, file, indent=4)



