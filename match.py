import time
import pandas as pd
import re
from ftfy import fix_text
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# INPUT FILE NAME
INPUTFILE = '/content/data.xlsx'
ITEMS_TAB = 'items'
ITEMS_HEADER = 'name'
MASTER_TAB = 'master'
MASTER_HEADER = 'name'

# Create ngrams
def ngrams(string, n=3):
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# Co-sine distance matching
def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

# Matching query
def getNearestN(query):
  queryTFIDF_ = vectorizer.transform(query)
  distances, indices = nbrs.kneighbors(queryTFIDF_)
  return distances, indices

if __name__ == '__main__':

    t1 = time.time()
	
    # Read data
    xls = pd.ExcelFile(INPUTFILE)
    df_items = pd.read_excel(xls, ITEMS_TAB, index_col=False)
    df_items = df_items.fillna('')
    itemlist = df_items[ITEMS_HEADER]

    df_master = pd.read_excel(xls, MASTER_TAB, index_col=False)
    masterlist = df_master[MASTER_HEADER]
    #print(masterlist)

	# Vectorize
    print('Vectorizing the data - this could take a few minutes for large datasets...')
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tfidf = vectorizer.fit_transform(masterlist.values.astype('U')) #masterlist
    print('Vectorizing completed...')
	
	# Nearest N
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    items = set(itemlist)
    print('Getting nearest N...')
    distances, indices = getNearestN(items)
    t = time.time()-t1
    print('Completed in:', round(t,2), 'secs')
	
    # dictionary of matches
    match_dict = {}

	# Get Matches
    items = list(items)
    print('Finding matches...')
    matches = []
    for i,j in enumerate(indices):
        # add to match dictionary
        match_dict[items[i]] = {'match': masterlist[j].values[0], 'score': round(distances[i][0],2)}

  	# Build dataframe and save to CSV
    print('Building data frame...')
    
    # construct matched dataset
    for index, row in df_items.iterrows():
        item = row[ITEMS_HEADER]

        match_obj = match_dict[item]
        
        # check if match or no match
        if match_obj:
            temp = [item, match_obj['match'], match_obj['score']]
        else:
            temp = [item, 'no-match', '-1']
        
        matches.append(temp)

    matches = pd.DataFrame(matches, columns=['Original name','Matched name', 'Match confidence (lower is better)'])
    matches.to_csv('matched-' + ITEMS_TAB + '.csv', index=False)
    print('Output recorded in', 'matched-' + ITEMS_TAB + '.csv')
    print('Done')