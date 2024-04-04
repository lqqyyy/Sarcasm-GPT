import json
import random
import requests

def expand_and_predict(original_data, url, max_context_length=50):
    sarcasm_to_emotion = {True: "ironic", False: "not ironic"}
    TP, FP, TN, FN = 0, 0, 0, 0

    for key, data in original_data.items():
        full_dialogue_length = len(data["context"])
        context_length = min(full_dialogue_length, max_context_length)

        votes = {"ironic": 0, "not ironic": 0}

        for _ in range(3):  # Voat
            for end in range(context_length, context_length + 1):
                # end = 5

                # selected_context = data["context"][-end:]
                # selected_speakers = data["context_speakers"][-end:]

                selected_context = []
                selected_speakers = []

                # print(len(data["context"]), end)
                if len(data["context"]) < end:
                    selected_context = data["context"][:]
                    selected_speakers = data["context_speakers"][:]
                else:
                    selected_context = data["context"][-end:].copy()
                    selected_speakers = data["context_speakers"][-end:].copy()

                selected_context.append(data['utterance'])
                selected_speakers.append(data['speaker'])


                context_dialogue = " \n".join([f'Speaker_{speaker}: "{line}"' for speaker, line in zip(selected_speakers, selected_context)])
                # print(context_dialogue)



                example_decision = random.choice([True, False])
                with open("sarcasm_data_origin_true.json", 'r', encoding='utf-8') as file:
                    sarcasm_data_origin_true = json.load(file)
                with open("sarcasm_data_origin_false.json", 'r', encoding='utf-8') as file:
                    sarcasm_data_origin_false = json.load(file)

                def select_last_lines(dialogue, end):
                    return ' \n'.join(dialogue.split('\n')[-end:])

                example1_dialogue = select_last_lines(random.choice(sarcasm_data_origin_true), end)
                example2_dialogue = select_last_lines(random.choice(sarcasm_data_origin_false), end)
                example3_dialogue = select_last_lines(random.choice(sarcasm_data_origin_true), end)
                example4_dialogue = select_last_lines(random.choice(sarcasm_data_origin_false), end)

                examples = "".join([
                    f"Example 1:\nDialogue: {example1_dialogue}\nClassification: ironic\n\n",
                    f"Example 2:\nDialogue: {example2_dialogue}\nClassification: not ironic\n\n",
                    f"Example 3:\nDialogue: {example3_dialogue}\nClassification: ironic\n\n",
                    f"Example 4:\nDialogue: {example4_dialogue}\nClassification: not ironic\n\n"
                ])

                # examples = ""
                instruction = f"Below is a dialogue between multiple people, in context, is the last sentence ironic?\n{examples}Dialogue: {context_dialogue}\nClassification: "
                # print(instruction)
                post_data = {"prompt": instruction, "history": []}

                response = requests.post(url, headers={'Content-Type': 'application/json'}, json=post_data)
                response_data = response.json()

                if response_data.get("detail") is not None:
                    continue  # 错误处理，简化版

                server_output = response_data.get("response").replace("</s>", "").lower()
                if "not ironic" in server_output:
                    server_output = "not ironic"
                else:
                    server_output = "ironic"

                votes[server_output] += 1

        print(votes)
        final_decision = max(votes, key=votes.get)
        correct_answer = sarcasm_to_emotion[data["sarcasm"]]

        if final_decision == correct_answer:
            if final_decision == "ironic":
                TP += 1
            else:
                TN += 1
        else:
            if final_decision == "ironic":
                FP += 1
            else:
                FN += 1


    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else 0
    f1_score = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0

    return accuracy, precision, recall, f1_score


input_file_path = 'sarcasm_data_test.json'
url = "http://0.0.0.0:8000/stream"

with open(input_file_path, 'r', encoding='utf-8') as file:
    original_data = json.load(file)


accuracy, precision, recall, f1_score = expand_and_predict(original_data, url)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1_score:.4f}")
