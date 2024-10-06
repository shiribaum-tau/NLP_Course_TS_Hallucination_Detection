from factscore.factscorer import FactScorer
import nltk
import os
import glob
import pickle
import json

nltk.download('punkt_tab')

KEY = ".key"
DATA_ROOT = os.path.join("..", "gen_data") #"/home/joberant/NLP_2324b/kr/output"

def remove_prefix(s, pref):
    if s.startswith(pref):
        return s[len(pref):]
    return s

def main():
    fs = FactScorer(openai_key=KEY)
    dat_files = glob.glob(os.path.join(DATA_ROOT, "*.pkl"))
    out_data = []
    for dat_file in dat_files:
        with open(dat_file, "rb") as f:
            generation_data = pickle.load(f)
        stripped_generation = remove_prefix(generation_data['generated_text'],
                                            generation_data['prompt'])
        out_data.append(dict(input=generation_data['prompt'],
                             output=stripped_generation,
                             topic=generation_data['entity'],
                             cat=["N/A", "N/A"]))
    topics = [i['topic'] for i in out_data]
    generations = [i['output'] for i in out_data]

    checked_facts = fs.get_score(topics, generations, gamma=10)
    # Now iterate over the topics
    for out_dat, decisions in zip(out_data, checked_facts['decisions']):
        annotations = []
        sent_to_atoms = checked_facts['sentences_to_facts'][out_dat['topic']]
        atom_to_verdict = {i['atom']: i['is_supported'] for i in decisions}
        for sentence, atoms in sent_to_atoms:
            annot = dict(text=sentence)
            annot['is-relevant'] = True
            annot['human-atomic-facts'] = [{ "text": i, "label": "S" if atom_to_verdict[i] else "NS" } for i in atoms]
            annotations.append(annot)
        out_dat['annotations'] = annotations

    with open("output.json", "w") as f:
        json.dump(out_data, f)

# import ipdb;ipdb.set_trace()
# print(out)
# # topics = ["Amelia Earhart"]
# # generations = ["Amelia Earhart (1897-1937) was a pioneering American aviator, author, and one of the most celebrated female pilots in history. She became the first woman to fly solo across the Atlantic Ocean in 1932, a feat that cemented her status as a trailblazer in aviation. Earhart was born in Atchison, Kansas, and developed a passion for flying after her first airplane ride in 1920. She soon set numerous aviation records and was an advocate for women in aviation, encouraging others to pursue careers in a male-dominated field. In addition to her flying accomplishments, Earhart was a public speaker, writer, and editor, using her platform to promote aviation and gender equality. She co-founded the Ninety-Nines, an organization for female pilots, and became a member of Purdue University's aviation faculty. In 1937, while attempting to become the first person to fly around the world along the equator, Earhart and her navigator, Fred Noonan, disappeared over the Pacific Ocean, and their fate remains one of the greatest unsolved mysteries in aviation history."]
# topics = ["Margaret Rose Vendryes"]
# generations = ["Margaret Rose Vendryes is an accomplished visual artist, art historian, and educator. Born in Kingston, Jamaica, she later moved to the United States and pursued her passion for art. She earned her Bachelor of Arts degree in Fine Arts and French from Amherst College in 1981. In 1991, she completed her Master of Arts degree in Art History from Tulane University, and in 1997, she obtained her Ph.D. in Art History from Princeton University.\n\nVendryes' work as a visual artist is characterized by her unique approach to combining traditional African mask motifs with contemporary pop culture icons, particularly focusing on African-American female subjects. Her African Diva Project, a series of paintings, is an excellent example of her artistic vision, which explores themes of identity, representation, and empowerment.\n\nAs an art historian, Vendryes has focused on African and African-American art, making significant contributions to the field through her research and publications. She has written articles and essays for various journals, edited volumes, and exhibition catalogs. Her book, \"Barthe: A Life in Sculpture,\" is a comprehensive study of the life and work of African-American sculptor Richmond Barth\u00e9, published in 2008.\n\nVendryes has also devoted her career to education. She has held teaching positions at various institutions, including York College (CUNY), where she served as Chair of the Department of Performing and Fine Arts from 2012 to 2019. In addition, she has curated multiple art exhibitions and given numerous lectures and presentations throughout her career.\n\nMargaret Rose Vendryes' multifaceted work as an artist, scholar, and educator has made her a prominent figure in both the art world and academia. Her unique perspective and style have left a lasting impact on contemporary art and the study of African and African-American art history."]
# # topics = ["Patrick Merrill"]
# # generations = ["Patrick Merrill is an American actor known for his work in film, television, and stage. He has appeared in numerous films, including The Dark Knight, The Fighter, and The Town, and has also performed in numerous television shows, including Law & Order: SVU, The Good Wife, and The Blacklist. He has won numerous awards and accolades for his work, including a Tony Award for his performance in The Adventures of Tom Sawyer and a Grammy Award for his work on the album Songs for a Dark Road.\n\nMerrill was born in 1962 in Scranton, Pennsylvania, and moved with his family to New York City at a young age. He studied acting at the University of Connecticut and began his career in the theater, performing in plays such as Cabaret and The Iceman Cometh. He made his film debut in the 1992 comedy The Princess Bride, and went on to appear in numerous films and television shows, including The Shawshank Redemption, Pulp Fiction, and The Fighter.\n\nIn addition to his work in film and television, Merrill has also directed and choreographed numerous plays and musicals. He is widely regarded as one of the most talented and versatile actors of his generation, and continues to work in both film and theater, making a significant impact on the industry."]



if __name__ == "__main__":
    main()