import os
import numbers
import decimal
import operator
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np


class AwesomeTextParserThatIWrote:
    PUNCT_REGEX = re.compile('[%s]' % re.escape("""!"#%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    ADJECTIVE_TAGS = ['jj', 'jjr', 'jjs']
    ADVERB_TAGS = ['rb', 'rbr', 'rbs', 'wrb']
    NOUN_TAGS = ['nn', 'nns', 'nnp', 'nnps']
    PRONOUN_TAGS = ['prp', 'prp$', 'wp', 'wp$']
    VERB_TAGS = ['vb', 'vbd', 'vbg', 'vbn', 'vbp', 'vbz']
    STOP_WORDS = set(stopwords.words('english'))

    def __init__(self, verbose=False):
        self._file_path = ''
        self._file = None
        self._dir = ''
        self._tagged_file = None
        self.title = []
        self.language = ''
        self.tags = []
        self.content = []
        self.punctuation = []
        self.data = []
        self.verbose = verbose
        self.languages = []
        self.common_stop_words = []
        self.stop_words = {}
        self.common_punctuation = []

    def parse_directory(self, dir):
        self._dir = dir
        for filename in os.listdir(self._dir):
            if filename.endswith('.txt'):
                self.data.append(self.parse_file(filename))

    def parse_file(self, filename):
        self._file_path = filename
        if self._dir is not '':
            self._file = open(os.path.join(self._dir, self._file_path), 'r')
        else:
            self._file = open(self._file_path, 'r')
        
        self.strip_title()
        self.content = self._file.read().lower()

        if self._dir is not '':
            f = re.sub(r'\'|\"|\;|\?|\:', '_', self._file_path)
            tagged_filename = os.path.join('tagged-corpus', f)
        else:
            # self._tagged_file = open('tagged-corpus/Aesop_BROTHER AND SISTER.txt')
            tagged_filename = 'tagged-corpus/Aesop_BROTHER AND SISTER.txt'
            print('\n\n!!!!--USING TEST TAG FILE--!!!!')
        
        # print(tagged_filename)
        self.tags = self.parse_tag_file(tagged_filename)

        tokens = nltk.word_tokenize(self.content)
        self.punctuation = []
        punctuation = [p for p in tokens if (not p.isalpha() and p != '.')]
        for p in punctuation:
            self.punctuation.append(re.sub(r'[A-Za-z]+', '', p))

        char_count = self.get_char_count(self.content)
        word_count = self.get_word_count(self.content)
        avg_word_length = self.get_avg_word_length(char_count, word_count)
        title_length = len(self.title)
        # most_common_punctuation = self.get_most_common_punctuation(self.punctuation)

        words = self.PUNCT_REGEX.sub('', self.content).split()
        most_common_stop_word = self.get_most_common_stop_word(words)
        
        self._file.close()

        data = [
            char_count, word_count, avg_word_length,
            most_common_stop_word, title_length, # most_common_punctuation,
        ]
        
        if most_common_stop_word not in self.common_stop_words:
            self.common_stop_words.append(most_common_stop_word)
        
        for tag in self.tags.items():
            data.append(tag[1])

        language = self.get_language(filename)
        data.append(language)
        if len(data) != 11:
            print('Missing data for file: {}'.format(self.title))
            if char_count is None:
                print('Missing char_count')
            elif word_count is None:
                print('Missing word_count')
            elif avg_word_length is None:
                print('Missing avg_word_length')
            elif title_length is None:
                print('Missing title_length')
            elif most_common_stop_word is None:
                print('Missing most_common_stop_word')
            else:
                print('Tags:', self.tags)
        if language not in self.languages:
            self.languages.append(language)
        return data

    def parse_tag_file(self, filename):
        self._tagged_file = open(filename, 'r')
        content = self._tagged_file.read().lower().split()
        tags = {
            'adjective': 0,
            'adverb': 0,
            'noun': 0,
            'pronoun': 0,
            'verb': 0
        }
        i = 0
        while i < len(content):
            word = self.PUNCT_REGEX.sub('', content[i])
            tag = self.PUNCT_REGEX.sub('', content[i+1])
            if word == '':
                i += 2
                continue

            if not self.verbose:
                tag = self.get_tag_type(tag)
            if tag != '':
                tags[tag] += 1

            i += 2

        for key in tags.keys():
            tags[key] /= len(content)

        return tags

    def strip_title(self):
        title = self._file.readline().lower()
        title = self.PUNCT_REGEX.sub('', title)
        self.title = title.split()
        self._file.readline()
        self._file.readline()

    def get_char_count(self, content):
        mashed = re.sub(r'\s', '', content)
        return len(mashed)

    def get_word_count(self, content):
        words = content.split()
        return len(words)

    def get_most_common_punctuation(self, punctuation):
        most_common = ''
        most_common_count = -1
        punct_types = {}
        for p in punctuation:
            if p not in punct_types:
                punct_types[p] = 0
            punct_types[p] += 1
            
            if most_common_count < punct_types[p]:
                most_common = p
                most_common_count = punct_types[p]
        if most_common not in self.common_punctuation:
            self.common_punctuation.append(most_common)
        return most_common

    def get_language(self, filename):
        language = filename.split('_')[0]
        cleaned = re.sub(r'\(|\-|\)', '', language)
        if cleaned == 'EnglishLancashire':
            cleaned = 'English'
        elif cleaned == 'CollectionIndianStories':
            cleaned = 'Indian'
        return cleaned.lower()

    def get_tag_counts(self, tags):
        adjective_count = 0
        adverb_count = 0
        noun_count = 0
        pronoun_count = 0
        verb_count = 0
        for t in tags:
            if t[1].startswith('j'):
                adjective_count += 1
            elif t[1].startswith('r') or t[1] == 'wrb':
                adverb_count += 1
            elif t[1].startswith('n'):
                noun_count += 1
            elif t[1] in self.PRONOUN_TAGS:
                pronoun_count += 1
            elif t[1].startswith('v'):
                verb_count += 1

        return {
            'adjectives': adjective_count,
            'adverbs': adverb_count,
            'nouns': noun_count,
            'pronouns': pronoun_count,
            'verbs': verb_count
        }

    def get_verbose_tag_counts(self, tags):
        tag_types = {}
        for t in tags:
            if t[1] not in tag_types:
                tag_types[t[1]] = 0
            tag_types[t[1]] += 1
        return tag_types

    def get_avg_word_length(self, char_count, word_count):
        return char_count / word_count

    def get_tag_type(self, tag):
        t = ''
        if tag.startswith('j'):
            t = 'adjective'
        elif tag.startswith('r') or tag == 'wrb':
            t = 'adverb'
        elif tag.startswith('n'):
            t = 'noun'
        elif tag in self.PRONOUN_TAGS:
            t = 'pronoun'
        elif tag.startswith('v'):
            t = 'verb'
        return t

    def get_most_common_stop_word(self, words):
        stop_words = {}
        most_common_stop_word = ''
        top_stop_word_count = -1
        for word in words:
            if word not in self.STOP_WORDS:
                continue
            if word not in stop_words:
                stop_words[word] = 0
            stop_words[word] += 1
            if top_stop_word_count < stop_words[word]:
                most_common_stop_word = word
                top_stop_word_count = stop_words[word]
        return most_common_stop_word

    def write_output(self, out_filename):
        out = open('data/{}.arff'.format(out_filename), 'w')
        out.write('@relation {}\n'.format(out_filename))
        out.write('@attribute character_count real\n')
        out.write('@attribute word_count real\n')
        out.write('@attribute average_word_length real\n')

        out.write('@attribute most_common_stop_word {')
        for w in range(len(self.common_stop_words) - 1):
            out.write('{}, '.format(self.common_stop_words[w]))
        out.write('%s}\n' % self.common_stop_words[-1])

        out.write('@attribute title_word_count real\n')

        # out.write('@attribute most_common_punctuation {')
        # for p in range(len(self.common_punctuation) - 2):
        #     out.write('\{}, '.format(self.common_punctuation[p]))
        # out.write('\%s}\n' % self.common_punctuation[-1])

        if not self.verbose:
            out.write('@attribute adjective_percent real\n')
            out.write('@attribute adverb_percent real\n')
            out.write('@attribute noun_percent real\n')
            out.write('@attribute pronoun_percent real\n')
            out.write('@attribute verb_percent real\n')
        else:
            out.write('@attribute coordinating_conjunctions real\n')
            out.write('@attribute cardinal_digits real\n')
            out.write('@attribute determiners real\n')
            out.write('@attribute existential_theres real\n')
            out.write('@attribute foreign_words real\n')
            out.write('@attribute preposistion_subordinating_conjunctions real\n')
            out.write('@attribute adjectives real\n')
            out.write('@attribute comparative_adjectives real\n')
            out.write('@attribute superlative_adjectives real\n')
            out.write('@attribute list_markers real\n')
            out.write('@attribute modals real\n')
            out.write('@attribute singular_nouns real\n')
            out.write('@attribute plural_nouns real\n')
            out.write('@attribute singular_proper_nouns real\n')
            out.write('@attribute plural_proper_nouns real\n')
            out.write('@attribute predeterminers real\n')
            out.write('@attribute possessives real\n')
            out.write('@attribute personal_pronouns real\n')
            out.write('@attribute possessive_pronouns real\n')
            out.write('@attribute adverbs real\n')
            out.write('@attribute comparative_adverbs real\n')
            out.write('@attribute superlative_adverbs real\n')
            out.write('@attribute particles real\n')
            out.write('@attribute to_s real\n')
            out.write('@attribute interjections real\n')
            out.write('@attribute verbs real\n')
            out.write('@attribute base_verbs real\n')
            out.write('@attribute past_verbs real\n')
            out.write('@attribute present_participle_verbs real\n')
            out.write('@attribute past_participle_verbs real\n')
            out.write('@attribute singular_present_verbs real\n')
            out.write('@attribute third_person_singular_present_verbs real\n')
            out.write('@attribute determiners real\n')
            out.write('@attribute determiner_pronouns real\n')
            out.write('@attribute determiner_adverbs real\n')
        out.write('@attribute class {')
        for l in range(len(self.languages) - 1):
            out.write('{}, '.format(self.languages[l]))
        out.write('%s}\n' % self.languages[-1])
        out.write('@data\n')
        out.write('%\n% {} instances\n%\n'.format(len(self.data)))
        for row in self.data:
            if len(row) != 11:
                print('Row missing data:',row)
            # for entry in range(5):
            #     out.write('{},'.format(row[entry]))
            # out.write('\{},'.format(row[5]))
            # for entry in range(6, len(row) - 1):
            #     out.write('{},'.format(row[entry]))
            # out.write('{}\n'.format(row[-1]))
            for entry in range(len(row) - 1):
                out.write('{},'.format(row[entry]))
            out.write('{}\n'.format(row[-1]))
        out.write('%\n%\n%')
        out.close()
        


parser = AwesomeTextParserThatIWrote(verbose=False)
parser.parse_directory('corpus/separated')
parser.write_output('corpus_simple')

# parser.parse_file('corpus/separated/Aesop_BROTHER AND SISTER.txt')
