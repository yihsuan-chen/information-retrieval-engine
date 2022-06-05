# information-retrieval-engine
School assignment for information retrieval

## 1:Document Index
You could indexing your collection of txt documents by the following comment
```
python MySearchEngine.py --task index --input COLLECTION_DIR --output INDEX_OUTPUT_DIR
```

## 2: Query Processing
```
python MySearchEngine.py --task search --input INDEX_OUTPUT_DIR --output QUERY_OUTPUT_DIR --k TOPK-DOCUMENTS --keyword WORD1 WORD2...


 python MySearchEngine.py --task search_p --input result --n 3 --output output --keyword fish world star
```

