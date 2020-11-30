
Datasets are derived from CogDL
=================
Autograph now supports the following benchmarks for different tasks:
- semi-supervised node classification: Cora, Citeseer, Pubmed, Amazon Computers\*, Amazon Photo\*, Coauthor CS\*, Coauthor Physics\*, Reddit （\*: using `utils.random_splits_mask_class` for splitting dataset is recommended.)


|  Dataset  |  PyG  |  CogDL  | x | y | edge_index | edge_attr | train/val/test node | train/val/test mask |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| Cora | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  | ✓ |
| Citeseer | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  | ✓ |
| Pubmed | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  | ✓ |
| Amazon Computers | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  |  |
| Amazon Photo | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  |  |
| Coauthor CS | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  |  |
| Coauthor Physics | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  |  |
| Reddit | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  | ✓ |

---

TODO:
Autograph now supports the following benchmarks for different tasks:
- unsupervised node classification: PPI, Blogcatalog, Wikipedia
- semi-supervised node classification: Cora, Citeseer, Pubmed
- heterogeneous node classification: DBLP, ACM, IMDB
- link prediction: PPI, Wikipedia, Blogcatalog
- multiplex link prediction: Amazon, YouTube, Twitter
- unsupervised graph classification: MUTAG, IMDB-B, IMDB-M, PROTEINS, COLLAB
- supervised graph classification: MUTAG, IMDB-B, IMDB-M, PROTEINS, COLLAB


|  Dataset  |  PyG  |  CogDL  | x | y | edge_index | edge_attr | train/val/test node | train/val/test mask | adj|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  ACM  |  |  ✓  |  ✓  | ✓ |  ✓  |    | ✓ |    | ✓ list |
|  DBLP |  |  ✓  |  ✓  | ✓ |  ✓  |    | ✓ |    | ✓ list |
|  IMDB |  |  ✓  |  ✓  | ✓ |  ✓  |    | ✓ |    | ✓ list |
| Flickr |  |  ✓  |    | ✓ |  ✓  |  ✓  |  |    |    |
| Blogcatalog |  |  ✓  |    | ✓ |  ✓  |  ✓  |  |    |    |
| PPI |  |  ✓  |    | ✓ |  ✓  |  ✓  |  |    |    |
| Wikipedia |  |  ✓  |    | ✓ |  ✓  |  ✓  |  |    |    |
| Amazon |  |  ✓  |    |  |    |    | ✓ data |    |    |
| Twitter |  |  ✓  |    |  |    |    | ✓ data |    |    |
| Youtube |  |  ✓  |    |  |    |    | ✓ data |    |    |
| Cora | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  | ✓ |    |
| Citeseer | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  | ✓ |    |
| Pubmed | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  | ✓ |    |
| Reddit | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  | ✓ |    |
| Mutag | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  |    |    |
| IMDB-B | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| IMDB-M | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| PROTEINS | ✓ |    |  ✓  | ✓ | ✓   |    |  |    |    |
| COLLAB | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| Reddit-B | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| Reddit-Multi-5k | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| Reddit-Multi-12k | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| PTC-MR | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  |    |    |
| NCI1 | ✓ |    |  ✓  | ✓ |  ✓  |    |  |    |    |
| NCI109 | ✓ |    |  ✓  | ✓ |  ✓  |    |  |    |    |
| Enzyme | ✓ |    |  ✓  | ✓ |  ✓  |    |  |    |    |








