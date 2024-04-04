import copy
import json
from openai import OpenAI

# 假设的 process_dialogue 函数
def process_dialogue(dialogue):
    # 这里是处理对话的逻辑，返回处理后的字符串
    # 为了演示，我们假设它只是简单地返回了同样的对话

    formatted_dialogue = format_dialogue(dialogue)
    explain = get_reason(formatted_dialogue)
    # print(explain)
    return explain

def get_reason(content):
    client = OpenAI(api_key="")

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
    sarcasm = True

    # Format the dialogue
    formatted_dialogue = "#### \nDialogue:\n"
    # Add the last utterance
    formatted_dialogue += data

    # Add the analysis request
    sarcasm_label = "<ironic>" if sarcasm else "<not ironic>"
    formatted_dialogue += "####Please briefly explain why the last sentence in the above dialogue is ironic, and does not require a repetition of the original text. Your explanation should be about 50 words"


    return formatted_dialogue


def main():
    with open('data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_data = []
    processed_data_copy = []

    for item in data:
        item_ = copy.deepcopy(item)

        if item["output"] == "ironic":
            dialogues = item["instruction"].split("Dialogue: ")[1:]
            last_dialogue = dialogues[-1].split("\nClassification:")[0]  # 去除最后的分类部分
            processed_dialogue = process_dialogue(last_dialogue)


            item_["output"] = "Explain: " + processed_dialogue + "Classification: ironic"
            item["output"] = "ironic. Explain: " + processed_dialogue


        processed_data.append(item)
        processed_data_copy.append(item_)


        with open('data-processed.json', 'w', encoding='utf-8') as file:
            json.dump(processed_data, file, ensure_ascii=False, indent=4)

        with open('data-processed-copy.json', 'w', encoding='utf-8') as file:
            json.dump(processed_data_copy, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()