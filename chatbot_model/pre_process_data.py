import os
import re

CURRENT_DIR = os.path.dirname(__file__)


def process_input_files(input_human, input_robot):
    data_path = f'{CURRENT_DIR}/{input_human}'
    data_path2 = f'{CURRENT_DIR}/{input_robot}'
    # Defining lines as a list of each line
    with open(data_path, 'r', encoding='UTF-8') as f:
        lines = f.read().split('\n')
    with open(data_path2, 'r', encoding='UTF-8') as f:
        lines2 = f.read().split('\n')
    lines = [re.sub(r"\[\w+\]", 'hi', line) for line in lines]
    lines = [" ".join(re.findall(r"\w+", line)) for line in lines]
    lines2 = [re.sub(r"\[\w+\]", '', line) for line in lines2]
    lines2 = [" ".join(re.findall(r"\w+", line)) for line in lines2]
    # grouping lines by response pair
    pairs = list(zip(lines, lines2))

    input_docs = []
    target_docs = []
    input_tokens = set()
    target_tokens = set()
    for line in pairs:
        input_doc, target_doc = line[0], line[1]
        # Appending each input sentence to input_docs
        input_docs.append(input_doc)
        # Splitting words from punctuation
        target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
        # Redefine target_doc below and append it to target_docs
        target_doc = '<START> ' + target_doc + ' <END>'
        target_docs.append(target_doc)

        # Now we split up each sentence into words and add each unique word to our vocabulary set
        for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
            if token not in input_tokens:
                input_tokens.add(token)
        for token in target_doc.split():
            if token not in target_tokens:
                target_tokens.add(token)

    return input_docs, target_docs, input_tokens, target_tokens


if __name__ == "__main__":
    dataset_human = 'human_text.txt'
    dataset_robot = 'robot_text.txt'
    input_docs, target_docs, input_tokens, target_tokens = process_input_files(dataset_human,
                                                                               dataset_robot)
    input_docs_casted = ','.join(input_docs)
    target_docs_casted = ','.join(target_docs)
    data_processed = open(f'{CURRENT_DIR}/data_processed.txt', 'w')
    data_processed.write(
        f'{input_docs_casted}\n{target_docs_casted}\n{input_tokens}\n{target_tokens}')
    data_processed.close()
