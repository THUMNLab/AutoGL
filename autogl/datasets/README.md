
Datasets are derived from PyG, OGB and CogDL.
=================
AutoGL now supports the following benchmarks for different tasks:
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


- supervised graph classification: MUTAG, IMDB-B, IMDB-M, PROTEINS, COLLAB

|  Dataset  |  PyG  |  CogDL  | x | y | edge_index | edge_attr | train/val/test node | train/val/test mask | adj|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| Mutag | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  |    |    |
| IMDB-B | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| IMDB-M | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| PROTEINS | ✓ |    |  ✓  | ✓ | ✓   |    |  |    |    |
| COLLAB | ✓ |    |    | ✓ | ✓   |    |  |    |    |

- node classification datasets from OGB: ogbn-products, ogbn-proteins, ogbn-arxiv, ogbn-papers100M and ogbn-mag.

- graph classification datasets from OGB: ogbg-molhiv, ogbg-molpcba, ogbg-ppa and ogbg-code.

---

TODO:
In future version, AutoGL will support the following benchmarks for different tasks:
- unsupervised node classification: PPI, Blogcatalog, Wikipedia
- heterogeneous node classification: DBLP, ACM, IMDB
- link prediction: PPI, Wikipedia, Blogcatalog
- multiplex link prediction: Amazon, YouTube, Twitter
- link prediction datasets from OGB: ogbl-ppa, ogbl-collab, ogbl-ddi, ogbl-citation, ogbl-wikikg and ogbl-biokg.

<!--
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
| NCI1 | ✓ |    |  ✓  | ✓ |  ✓  |    |  |    |    |
| NCI109 | ✓ |    |  ✓  | ✓ |  ✓  |    |  |    |    |
| Enzyme | ✓ |    |  ✓  | ✓ |  ✓  |    |  |    |    |
| Reddit-B | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| Reddit-Multi-5k | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| Reddit-Multi-12k | ✓ |    |    | ✓ | ✓   |    |  |    |    |
| PTC-MR | ✓ |    |  ✓  | ✓ |  ✓  |  ✓  |  |    |    |
-->

