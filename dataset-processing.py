import gc
import json

# 和prompt1相同，只是把标签换了一种
import json
import random

import numpy as np
from transformers import BertModel, BertTokenizer, pipeline
from scipy.spatial.distance import cosine

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
feature_extraction = pipeline('feature-extraction', model='bert-base-uncased', tokenizer='bert-base-uncased')

print("loda done")


def get_embedding(text):
    embeddings = feature_extraction(text)
    embeddings = np.mean(np.squeeze(embeddings), axis=0)
    return embeddings


def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)
    # return 1 - cosine(embedding1.detach().numpy(), embedding2.detach().numpy())

def precompute_embeddings(dialogues):
    embeddings = {}
    for dialogue in dialogues:
        embeddings[dialogue] = get_embedding(dialogue)
        gc.collect()
    return embeddings

def find_top_similar_dialogues(target_embedding, embeddings, top_k=10):
    similarities = []
    for dialogue, embedding in embeddings.items():
        similarity = cosine_similarity(target_embedding, embedding)
        similarities.append((dialogue, similarity))
    top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [dialogue for dialogue, _ in top_similarities]


def expand_data_with_limit(original_data, max_context_length, sarcasm_data_origin_true, sarcasm_data_origin_false):
    sarcasm_to_emotion = {True: "ironic", False: "not ironic"}
    expanded_data_list = []

    embeddings_origin_true = precompute_embeddings(sarcasm_data_origin_true)
    embeddings_origin_false = precompute_embeddings(sarcasm_data_origin_false)


    for key, data in original_data.items():

        full_dialogue_length = len(data["context"])

        for end in range(1, full_dialogue_length + 1):

            # 选择上下文
            selected_context = data["context"][-end:]
            selected_speakers = data["context_speakers"][-end:]

            selected_context.append(data['utterance'])
            selected_speakers.append(data['speaker'])


            # if data['speaker'] not in selected_speakers:
            #     selected_speakers.append(data['speaker'])
            #     selected_context.append(data['utterance'])  # 可选：也可以将 utterance 添加到 context 中
            # speaker_id = f"Speaker_{selected_speakers.index(data['speaker'])}"


            # context_dialogue = " \t".join([f'Speaker_{i}: "{line}"' for i, (speaker, line) in enumerate(zip(selected_speakers, selected_context))])
            context_dialogue = " \n".join([f'Speaker_{speaker}: "{line}"' for i, (speaker, line) in
                                           enumerate(zip(selected_speakers, selected_context))])

            context_dialogue_embedding = get_embedding(context_dialogue)
            # example_dialogue = find_most_similar_dialogue(context_dialogue_embedding, embeddings_origin_true)

            top_similar_dialogues_true = find_top_similar_dialogues(context_dialogue_embedding, embeddings_origin_true)
            top_similar_dialogues_false = find_top_similar_dialogues(context_dialogue_embedding,
                                                                     embeddings_origin_false)
            example_dialogue_true = random.choice(top_similar_dialogues_true)
            example_dialogue_false = random.choice(top_similar_dialogues_false)


            example = (
                "Example 1:\n" +
                "Dialogue: " + example_dialogue_true.split("||")[0] + "\n" +
                "Classification: ironic. Explain: " + example_dialogue_true.split("||")[1] + "\n\n"
            )

            # example_dialogue = find_most_similar_dialogue(context_dialogue_embedding, embeddings_origin_false)
            # print("example_dialogue_false", example_dialogue)
            example = example + (
                "Example 2:\n" +
                "Dialogue: " + example_dialogue_false + "\n" +
                "Classification: not ironic\n\n"
            )

            example_dialogue_true = random.choice(top_similar_dialogues_true)
            example_dialogue_false = random.choice(top_similar_dialogues_false)


            example = example +(
                "Example 3:\n" +
                "Dialogue: " + example_dialogue_true.split("||")[0] + "\n" +
                "Classification: ironic. Explain: " + example_dialogue_true.split("||")[1] + "\n\n"
            )

            # example_dialogue = find_most_similar_dialogue(context_dialogue_embedding, embeddings_origin_false)
            # print("example_dialogue_false", example_dialogue)
            example = example + (
                "Example 4:\n" +
                "Dialogue: " + example_dialogue_false + "\n" +
                "Classification: not ironic.\n\n"
            )


            instruction = (
                "Below is a dialogue between multiple people, in context, is the last sentence ironic? \n" + example +
                "Dialogue: " + context_dialogue + "\n" +
                "Classification:"
            )

            if data["sarcasm"]:
                # 构建新的数据格式
                expanded_data = {
                    "instruction": instruction,
                    "input": "",
                    "output": "ironic."
                }
            else:
                expanded_data = {
                    "instruction": instruction,
                    "input": "",
                    "output": "not ironic."
                }

            expanded_data_list.append(expanded_data)

    # 打开JSON文件
    with open('output-exp-train-gpt.json', 'r') as file1:
        data = json.load(file1)

    for item in data:
        result = item.split("||")
        context_dialogue = result[0]
        explain = result[1]


        context_dialogue_embedding = get_embedding(context_dialogue)

        top_similar_dialogues_true = find_top_similar_dialogues(context_dialogue_embedding, embeddings_origin_true)
        top_similar_dialogues_false = find_top_similar_dialogues(context_dialogue_embedding,
                                                                 embeddings_origin_false)
        example_dialogue_true = random.choice(top_similar_dialogues_true)
        example_dialogue_false = random.choice(top_similar_dialogues_false)

        # example_dialogue = find_most_similar_dialogue(context_dialogue_embedding, embeddings_origin_true)
        # print("example_dialogue_true", example_dialogue)
        example = (
                "Example 1:\n"
                "Dialogue: " + example_dialogue_true.split("||")[0] + "\n" +
                "Classification: ironic. Explain: " + example_dialogue_true.split("||")[1] + "\n\n"
        )

        # example_dialogue = find_most_similar_dialogue(context_dialogue_embedding, embeddings_origin_false)
        # print("example_dialogue_false", example_dialogue)
        example = example + (
                "Example 2:\n"
                "Dialogue: " + example_dialogue_false + "\n" +
                "Classification: not ironic.\n\n"
        )

        example_dialogue_true = random.choice(top_similar_dialogues_true)
        example_dialogue_false = random.choice(top_similar_dialogues_false)
        example = example + (
                "Example 3:\n"
                "Dialogue: " + example_dialogue_true.split("||")[0] + "\n" +
                "Classification: ironic. Explain: " + example_dialogue_true.split("||")[1] + "\n\n"
        )

        # example_dialogue = find_most_similar_dialogue(context_dialogue_embedding, embeddings_origin_false)
        # print("example_dialogue_false", example_dialogue)
        example = example + (
                "Example 4:\n"
                "Dialogue: " + example_dialogue_false + "\n" +
                "Classification: not ironic.\n\n"
        )



        instruction = (
                # "Below is a dialogue between multiple people, in context, is the last sentence ironic? If it's ironic, explain why.\n" + example +
                "Below is a dialogue between multiple people, in context, is the last sentence ironic? \n" + example +
                "Dialogue: " + context_dialogue + "\n" +
                "Classification:"
        )

        # 构建新的数据格式
        expanded_data = {
            "instruction": instruction,
            "input": "",
            # "output": "ironic. Explain: " + explain
            "output": "ironic."
        }
        # print("2", explain)

        expanded_data_list.append(expanded_data)

    random.shuffle(expanded_data_list)
    return expanded_data_list



input_file_path = 'output-exp-train.json'  # 替换为实际文件路径
output_file_path = 'dataset-train.json'  # 替换为期望的输出文件路径

with open('data-processed.json', 'r', encoding='utf-8') as file:
    sarcasm_data_origin_true = json.load(file)

with open('sarcasm_data_origin_false.json', 'r', encoding='utf-8') as file:
    sarcasm_data_origin_false = json.load(file)

# 读取原始数据
with open(input_file_path, 'r', encoding='utf-8') as file:
    original_data = json.load(file)

# 处理数据
transformed_data_list = expand_data_with_limit(original_data, 50, sarcasm_data_origin_true, sarcasm_data_origin_false)

# 将处理后的数据保存到新文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(transformed_data_list, file, indent=4)

# 输出结果，以便检查
# print(json.dumps(transformed_data_list, indent=4))
