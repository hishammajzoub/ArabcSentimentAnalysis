# Note: Make sure that openpyxl is installed before running this script.
#   pip install openpyxl

# Normalizing Text

# This file contains a function that will serve as normalization in the Sentiment Analysis of
# Arabic Tweets Project. It contains an approach that uses Unicode to group the characters, words,
# and diacritics that we want to remove from the Arabic Tweet in order to normalize it.


# Importing the necessary libraries
import re
import string
import regex
from nltk.corpus import stopwords
import pandas as pd
import itertools

# Setting up the needed unicode characters

# Punctuation in Unicode, No Exclamation Mark in Arabic Unicode
COMMA = u'\u060C'
SEMICOLON = u'\u061B'
QUESTION = u'\u061F'
FULL_STOP = u'\u06d4'

# Letters needed for exceptions
HAMZA = u'\u0621'
ALEF_MADDA = u'\u0622'
ALEF_HAMZA_ABOVE = u'\u0623'
WAW_HAMZA = u'\u0624'
ALEF_HAMZA_BELOW = u'\u0625'
YEH_HAMZA = u'\u0626'
ALEF = u'\u0627'
TEH_MARBUTA = u'\u0629'
TATWEEL = u'\u0640'
MADDA_ABOVE = u'\u0653'
HAMZA_ABOVE = u'\u0654'
HAMZA_BELOW = u'\u0655'
ALEF_MAKSURA = u'\u0649'
YEH = u'\u064a'
HEH = u'\u0647'
LAM = u'\u0644'

# Symbols
PERCENT = u'\u066a'
DECIMAL = u'\u066b'
THOUSANDS = u'\u066c'
STAR = u'\u066d'
MULITIPLICATION_SIGN = u'\u00D7'
DIVISION_SIGN = u'\u00F7'
MINI_ALEF = u'\u0670'
ALEF_WASLA = u'\u0671'
BYTE_ORDER_MARK = u'\ufeff'

# Diacritics in Unicode
FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'

# Ligatures in Unicode
LAM_ALEF = u'\ufefb'
LAM_ALEF_HAMZA_ABOVE = u'\ufef7'
LAM_ALEF_HAMZA_BELOW = u'\ufef9'
LAM_ALEF_MADDA_ABOVE = u'\ufef5'
SIMPLE_LAM_ALEF = u'\u0644\u0627'
SIMPLE_LAM_ALEF_HAMZA_ABOVE = u'\u0644\u0623'
SIMPLE_LAM_ALEF_HAMZA_BELOW = u'\u0644\u0625'
SIMPLE_LAM_ALEF_MADDA_ABOVE = u'\u0644\u0622'

# Grouping the Harakat, Hamzat, Alefat, and Lamalefat in order to easily manipulate them later.

# Using regular expressions library to compile a regular expression pattern
HARAKAT = re.compile(u"[" + u"".join([FATHATAN, DAMMATAN, KASRATAN,
                                      FATHA, DAMMA, KASRA, SUKUN,
                                      SHADDA]) + u"]")
HAMZAT = re.compile(u"[" + u"".join([WAW_HAMZA, YEH_HAMZA]) + u"]")
ALEFAT = re.compile(u"[" + u"".join([ALEF_MADDA, ALEF_HAMZA_ABOVE,
                                     ALEF_HAMZA_BELOW, HAMZA_ABOVE,
                                     HAMZA_BELOW]) + u"]")
LAMALEFAT = re.compile(u"[" + u"".join([LAM_ALEF,
                                        LAM_ALEF_HAMZA_ABOVE,
                                        LAM_ALEF_HAMZA_BELOW,
                                        LAM_ALEF_MADDA_ABOVE]) + u"]")

# Converting Eastern Arabic Numbers to Western Arabic Numbers
WESTERN_ARABIC = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

EASTERN_ARABIC = [u'۰', u'۱', u'۲', u'۳', u'٤', u'۵', u'٦', u'۷', u'۸', u'۹']

eastern_to_western = {}
for i in range(len(EASTERN_ARABIC)):
    eastern_to_western[EASTERN_ARABIC[i]] = WESTERN_ARABIC[i]

# Joining String Punctuation and Arabic Punctuation
arabic_punctuations = COMMA + SEMICOLON + QUESTION + PERCENT + DECIMAL + THOUSANDS + STAR + FULL_STOP + MULITIPLICATION_SIGN + DIVISION_SIGN
all_punctuations = string.punctuation + arabic_punctuations + '()[]{}'

all_punctuations = ''.join(list(set(all_punctuations)))


# Function to remove non-Arabic characters
def remove_non_arabic(text):
    arabic_pattern = r'[\u0600-\u06FF\ufb50-\ufdff\ufe70-\ufeff\s]|[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF]+'  # Pattern to match Arabic letters and emojis

    # Find all matches of Arabic letters and emojis
    matches = re.findall(arabic_pattern, text)

    # Join the matches to form the final cleaned string
    cleaned_text = ''.join(matches)

    return cleaned_text


# Function to normalize Hamza in text
def hamza(text):
    text = ALEFAT.sub(ALEF, text)
    return HAMZAT.sub(HAMZA, text)


# Function to normalize Teh Marbuta and Alef Maksura
def spellerrors(text):
    text = re.sub(u'[%s]' % TEH_MARBUTA, HEH, text)
    return re.sub(u'[%s]' % ALEF_MAKSURA, YEH, text)


# Function to remove underscore and replace it by a space
def remove_underscore(text):
    return text.replace("_", " ")


# Function to normalize LamAlef in text
def lamalef(text):
    return LAMALEFAT.sub(u'%s%s' % (LAM, ALEF), text)


# Function to remove retweet tags from text
def remove_retweet_tag(text):
    return re.compile('\#').sub('', re.compile('rt @[a-zA-Z0-9_]+:|@[a-zA-Z0-9_]+').sub('', text).strip())


# Function to remove emails from tweet
def replace_emails(text):
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    for email in emails:
        text = text.replace(email, '')
        # text = text.replace(email,' hasEmailAddress ')
    return text


# Function to remove URLs from text
def replace_urls(text):
    return re.sub(r"http\S+|www.\S+", "", text)


# Function to convert easter numerals to western numerals
def convert_eastern_to_western(text):
    for num in EASTERN_ARABIC:
        text = text.replace(num, eastern_to_western[num])
    return text


# Function to remove phone numbers from text
def replace_phonenb(text):
    return re.sub(r'\d{10}', '', text)


# Function to remove extra spaces from text
def remove_extra_spaces(text):
    return ' '.join(text.split())


# Function to remove consecutive occurrences of Arabic Letters: ex:ههههههههه becomes ه
def repeated_consecutively(text):
    return ''.join(c[0] for c in itertools.groupby(text))


# Function to insert a space between emojis and arabic words to consider the emoji
# a separate word when training
def separate_emojis_arabic_letters(text):
    pattern = r'(\p{Arabic}+)(\p{So}+)'
    separated_text = regex.sub(pattern, r'\1 \2 ', text)
    return separated_text


# Punctuations
punctuations = '''`÷×؛<=>\()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ#$+'''

# Setting stop words from NLTK Library
stop_words = stopwords.words()

# Setting arabic diacritics
arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


# Alternative Function that normalizes text in a slightly different way.
def normalize_tweet(text):
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)

    text = re.sub(arabic_diacritics, '', text)

    text = hamza(text)

    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    text = ' '.join(word for word in text.split() if word not in stop_words)

    text = lamalef(text)
    text = spellerrors(text)
    text = remove_retweet_tag(text)
    text = replace_emails(text)
    text = remove_underscore(text)
    text = replace_phonenb(text)
    text = replace_urls(text)
    text = convert_eastern_to_western(text)
    text = remove_non_arabic(text)
    text = separate_emojis_arabic_letters(text)
    text = remove_extra_spaces(text)
    text = repeated_consecutively(text)

    return text

# normalize a dataset
def normalize_dataset(file_name):
    data = pd.read_excel(file_name, header=None, names=["Text", "Sentiment"])
    data = data.dropna()
    data = data[data.Sentiment != "OBJ"]
    data["Text"] = data["Text"].apply(normalize_tweet)
    data.to_excel("Normalized " + file_name, index=False)

# start normalization message
print("\n Normalizing the 'Tweets.xlsx' dataset. Please wait...")

# normalize small tweets
normalize_dataset("Tweets.xlsx")

# normalization done
print("\n Normalization Done!")
