
with open("example_batch.jsonl", "rb") as f:
    ex = f.read().strip()

with open("all_entites.txt", "rb") as f:
    entites = f.readlines()

entites = [i.strip() for i in entites]

data = [ex.replace(b"bobob", person) for person in entites]

data = [byte.decode('utf-8') for byte in data]

joined_data = "\n".join(data)

with open('big_bio_batch.jsonl', 'w', encoding='utf-8') as file:
    file.write(joined_data)