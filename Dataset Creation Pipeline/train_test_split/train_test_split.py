#from sklearn.model_selection import train_test_split
import os
import shutil


# Divide the entities file
# def divide_entities():
#     with open('all_entities_OPT.txt', 'r') as f:
#         entities = f.readlines()
#
#     entities_train, entities_test = train_test_split(entities, test_size=0.2, random_state=42)
#
#     with open('entities_train_opt.txt', 'w') as f:
#         f.writelines(entities_train)
#
#     with open('entities_test_opt.txt', 'w') as f:
#         f.writelines(entities_test)


def divide_files(source_dir, target_dir):

    with open('entities_test_openai.txt', 'r') as f:
        test_entities = f.readlines()

    files_to_move = [f"{entity.strip()}_data.pkl" for entity in test_entities]

    for filename in files_to_move:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            print(f"Moved: {filename}")
        else:
            print(f"File not found: {filename}")

    print("File moving process completed.")


if __name__ == '__main__':
    src_dir = "/home/joberant/NLP_2324b/ronimeamen/data/mini_GPT"
    trg_dir = "/home/joberant/NLP_2324b/ronimeamen/data/mini_GPT/Test"

    divide_files(src_dir, trg_dir)

