from factscore.factscorer import FactScorer
import nltk
nltk.download('punkt_tab')

KEY = ".key"

fs = FactScorer(openai_key=KEY)

topics = ["Amelia Earhart"]
generations = ["Amelia Earhart (1897–1937) was a pioneering American aviator, author, and one of the most celebrated female pilots in history. She became the first woman to fly solo across the Atlantic Ocean in 1932, a feat that cemented her status as a trailblazer in aviation. Earhart was born in Atchison, Kansas, and developed a passion for flying after her first airplane ride in 1920. She soon set numerous aviation records and was an advocate for women in aviation, encouraging others to pursue careers in a male-dominated field. In addition to her flying accomplishments, Earhart was a public speaker, writer, and editor, using her platform to promote aviation and gender equality. She co-founded the Ninety-Nines, an organization for female pilots, and became a member of Purdue University's aviation faculty. In 1937, while attempting to become the first person to fly around the world along the equator, Earhart and her navigator, Fred Noonan, disappeared over the Pacific Ocean, and their fate remains one of the greatest unsolved mysteries in aviation history."]

out = fs.get_score(topics, generations, gamma=10)
print(out)