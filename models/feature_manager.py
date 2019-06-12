import os

from lexicons.AFINN.get_afinn_features import load_afinn_lexicon
from lexicons.Bing_Liu_opinion_lex.get_BL_features import bing_liu
from lexicons.MPQA.get_mpqa_features import mpqa_lex
from lexicons.NRC_Emotion_Lexicon_v092.get_nrc_emolex_features import emolex
from lexicons.SemEval2015_English_Twitter_Lexicon.get_semeval2015_twitter_features import semeval15_lexicon
from lexicons.affect_features.get_affect_features import \
    load_affect_features, fix_affect_features
from lexicons.liwc.get_liwc_features import load_liwc_lex
from sys_config import BASE_DIR
from utils.generic import multiple_lexicon_merge


def load_LIWC():
    liwc_lex_str = load_liwc_lex()
    liwc_lex = {key: [float(i) for i in value] for key, value in
                liwc_lex_str.items()}

    feat_lengths_dict = {key: len(value) for key, value in liwc_lex.items()}
    feat_length = list(set(feat_lengths_dict.values()))[0]

    assert feat_length == feat_length, "Not all words in LIWC have 73 dimensions!"
    return liwc_lex


def load_affect():
    feat2idx, idx2feat, affect_features = load_affect_features()

    affect_features = fix_affect_features(affect_features)

    return affect_features

def load_afinn():
    lex = load_afinn_lexicon(os.path.join(BASE_DIR, 'lexicons', 'AFINN', 'AFINN-111.txt'))
    lexx = {}
    for key in lex:
        lexx[key] = [float(lex[key])]

    feat_lengths_dict = {key: len(value) for key, value in lexx.items()}
    feat_length = list(set(feat_lengths_dict.values()))[0]

    assert feat_length == feat_length, "Not all words in AFINN lex have 1 dimension!"
    return lexx

def load_semeval15():
    lex = semeval15_lexicon()
    lexx = {}
    for key in lex:
        lexx[key] = [float(lex[key])]
    feat_lengths_dict = {key: len(value) for key, value in lexx.items()}
    feat_length = list(set(feat_lengths_dict.values()))[0]

    assert feat_length == feat_length, "Not all words in SemEval15 lex have 1 dimension!"
    return lexx

def load_emolex():
    lex = emolex()
    feat_lengths_dict = {key: len(value) for key, value in lex.items()}
    feat_length = list(set(feat_lengths_dict.values()))[0]

    assert feat_length == feat_length, "Not all words in emolex have 19 dimensions!"
    return lex

def load_bing_liu():
    lex = bing_liu()
    feat_lengths_dict = {key: len(value) for key, value in lex.items()}
    feat_length = list(set(feat_lengths_dict.values()))[0]

    assert feat_length == feat_length, "Not all words in bing liu lex have 1 dimension!"
    return lex

def load_mpqa():
    lex = mpqa_lex()
    feat_lengths_dict = {key: len(value) for key, value in lex.items()}
    feat_length = list(set(feat_lengths_dict.values()))[0]

    assert feat_length == feat_length, "Not all words in MPQA lex have 4 dimensions!"
    return lex

def load_features(features):
    if not isinstance(features, list):
        features = [features]
    lexicons = []
    for name in features:
        if name == "LIWC":
            lexicons.append(load_LIWC())
        elif name == "affect":
            lexicons.append(load_affect())
        elif name == "afinn":
            lexicons.append(load_afinn())
        elif name == "semeval15":
            lexicons.append(load_semeval15())
        elif name == "emolex":
            lexicons.append(load_emolex())
        elif name == "bing_liu":
            lexicons.append(load_bing_liu())
        elif name == "mpqa":
            lexicons.append(load_mpqa())
        else:
            raise ValueError(f"The dataset:'{name}' is not yet supported!")

    final_lex = multiple_lexicon_merge(lexicons)
    feat_lengths_dict = {key: len(value) for key, value in final_lex.items()}
    feat_length = list(set(feat_lengths_dict.values()))[0]

    assert feat_length == feat_length, "Not all words in final lex have correct dimensions!"
    return final_lex, feat_length

def feature_selection(X, p):
    from sklearn.feature_selection import VarianceThreshold
    # X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    Y = sel.fit_transform(X)
    return Y, Y.shape[1]
