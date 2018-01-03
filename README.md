
Rui Wang, 2017-2018

Flask Webapp Using Machine Learning

# Amazon Review Sentiment Analysis

[Amazon product data](http://jmcauley.ucsd.edu/data/amazon/links.html) was provided by Julian McAuley, UCSD.

We cover the setiment analysis of 12 catergories of Amazon prodcuts. We list our results in test datasets using [xxxx model]().

| Category                   | Accuracy (%) |
|----------------------------|--------------|
| Automotive                 | 88.94        |
| Baby                       | 90.13        |
| Clothing shoes and Jewelry | 90.58        |
| Digital music              | 88.40        |
| Electronics                | 91.62        |
| Grocery and Gourmet        | 90.19        |
| Home and Kitchen           | 92.03        |
| Kindle store               | 92.67        |
| Pet supplies               | 89.20        |
| Sports and Outdoors        | 91.18        |
| Toys and Games             | 91.09        |
| Video games                | 88.74        |

Format is one-review-per-line in (loose) json. See examples below for further help reading the data.

Sample review:

```
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```
where

* reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
* asin - ID of the product, e.g. 0000013714
* reviewerName - name of the reviewer
* helpful - helpfulness rating of the review, e.g. 2/3
* reviewText - text of the review
* overall - rating of the product
* summary - summary of the review
* unixReviewTime - time of the review (unix time)
* reviewTime - time of the review (raw)

Julian McAuley provides the codes to read the data into a pandas data frame:

```python
import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Video_Games.json.gz')
```
