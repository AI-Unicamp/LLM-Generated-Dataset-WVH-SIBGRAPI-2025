import random
import re
from random import shuffle
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


class TextAugmentor:
    def __init__(self, prob_synonym=0.1, prob_insert=0, prob_swap=0.1, prob_delete=0, num_aug=4):
        """
        Initialize text augmentation module with probabilities.
        
        Args:
            prob_stopword (float): Probability of removing stopwords.
            prob_synonym (float): Probability of replacing words with synonyms.
            prob_insert (float): Probability of inserting words.
            prob_swap (float): Probability of swapping words.
            prob_delete (float): Probability of deleting words.
            num_aug (int): Number of augmented sentences per input.
        """
        self.prob_synonym = prob_synonym
        self.prob_insert = prob_insert
        self.prob_swap = prob_swap
        self.prob_delete = prob_delete
        self.num_aug = num_aug

        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
            'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
            'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
            'will', 'just', 'don', 'should', 'now'
        ])

    def get_only_chars(self, line):
        """Clean text by keeping only alphabetic characters and spaces."""
        clean_line = ""
        line = line.replace("-", " ") #replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        line = line.lower()

        for char in line:
            if char in '''qwertyuiopasdfghjklzxcvbnm ’',.!?''':
                clean_line += char
            else:
                clean_line += ' '

        clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
        return clean_line

    def get_synonyms(self, word):
        """Retrieve synonyms of a word from WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def synonym_replacement(self, words, n):
        """Replace `n` words with their synonyms."""
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                #print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n: #only replace up to n words
                break

        #this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def random_deletion(self, words, p):
        """Randomly delete words with probability `p`."""
        if len(words) == 1:
            return words

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return new_words

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

    def augment(self, sentence):
	
        sentence = self.get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']
        num_words = len(words)
        
        augmented_sentences = []
        num_new_per_technique = int(self.num_aug/4)+1

        #sr
        if (self.prob_synonym > 0):
            n_sr = max(1, int(self.prob_synonym*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.synonym_replacement(words, n_sr)
                augmented_sentences.append(' '.join(a_words))

        #ri
        if (self.prob_insert > 0):
            n_ri = max(1, int(self.prob_insert*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))

        #rs
        if (self.prob_swap > 0):
            n_rs = max(1, int(self.prob_swap*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))

        #rd
        if (self.prob_delete > 0):
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(words, self.prob_delete)
                augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [self.get_only_chars(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)

        #trim so that we have the desired number of augmented sentences
        if self.num_aug >= 1:
            augmented_sentences = augmented_sentences[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        #append the original sentence
        augmented_sentences.append(sentence)

        return augmented_sentences