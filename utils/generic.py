import numpy
import umap
from sklearn.decomposition import PCA


def merge_dicts(a, b):
    a.update({k: v for k, v in b.items() if k in a})
    return a


def number_h(num):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, 'Yi')


def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b


def dim_reduce(data_sets, n_components=2, method="PCA"):
    data = numpy.vstack(data_sets)
    splits = numpy.cumsum([0] + [len(x) for x in data_sets])
    if method == "PCA":
        reducer = PCA(random_state=20, n_components=n_components)
        embedding = reducer.fit_transform(data)
    elif method == "UMAP":
        reducer = umap.UMAP(random_state=20,
                            n_components=n_components,
                            min_dist=0.5)
        embedding = reducer.fit_transform(data)
    else:
        reducer_linear = PCA(random_state=20, n_components=50)
        linear_embedding = reducer_linear.fit_transform(data)
        reducer_nonlinear = umap.UMAP(random_state=20,
                                      n_components=n_components,
                                      min_dist=0.5)
        embedding = reducer_nonlinear.fit_transform(linear_embedding)

    return [embedding[start:stop] for start, stop in pairwise(splits)]


def lexicon_merge(lex1, lex2):
    # merge two lexiconss into one
    values_lex1 = numpy.array(list(lex1.values()))
    values_lex2 = numpy.array(list(lex2.values()))

    feat_lex1 = values_lex1.shape[1]
    feat_lex2 = values_lex2.shape[1]

    final_lex = {}
    final_lex.update(lex1)

    for word in lex2:
        if word in final_lex.keys():
            final_lex[word] = final_lex[word] + lex2[word]
        else:
            final_lex[word] = list(numpy.zeros(feat_lex1)) + lex2[word]
    only_lex1 = [key for key in lex1.keys() if not key in lex2.keys()]

    for word in only_lex1:
        final_lex[word] += list(numpy.zeros(feat_lex2))

    return final_lex

def multiple_lexicon_merge(lexicons):
    if len(lexicons) == 1:
        return lexicons[0]
    elif len(lexicons) == 2:
        return lexicon_merge(lexicons[0], lexicons[1])
    elif len(lexicons) == 3:
        lex_1_2 = lexicon_merge(lexicons[0], lexicons[1])
        return lexicon_merge(lex_1_2, lexicons[2])
    elif len(lexicons) == 4:
        lex_1_2 = lexicon_merge(lexicons[0], lexicons[1])
        lex_1_2_3 = lexicon_merge(lex_1_2, lexicons[2])
        return lexicon_merge(lex_1_2_3, lexicons[3])
    elif len(lexicons) == 5:
        lex_1_2 = lexicon_merge(lexicons[0], lexicons[1])
        lex_1_2_3 = lexicon_merge(lex_1_2, lexicons[2])
        lex_1_2_3_4 = lexicon_merge(lex_1_2_3, lexicons[3])
        return lexicon_merge(lex_1_2_3_4, lexicons[4])
    elif len(lexicons) == 6:
        lex_1_2 = lexicon_merge(lexicons[0], lexicons[1])
        lex_1_2_3 = lexicon_merge(lex_1_2, lexicons[2])
        lex_1_2_3_4 = lexicon_merge(lex_1_2_3, lexicons[3])
        lex_1_2_3_4_5 = lexicon_merge(lex_1_2_3_4, lexicons[4])
        return lexicon_merge(lex_1_2_3_4_5, lexicons[5])
    else:
        raise ValueError(f"Merging of '{len(lexicons)}' number of lexiconss"
                         f"is not yet supported!")