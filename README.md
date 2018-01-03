
Rui Wang, 2017-2018

Flask Webapp Using Machine Learning

# Amazon Review Sentiment Analysis

[Amazon product data](http://jmcauley.ucsd.edu/data/amazon/links.html) was provided by Julian McAuley, UCSD.

We cover the setiment analysis of 12 catergories of Amazon prodcuts.

|--------------|---------------------|----------------------------|---------------|
| Automotive   | Baby                | Clothing shoes and Jewelry | Digital music |
| Electronics  | Grocery and Gourmet | Home and Kitchen           | Kindle store  |
| Pet supplies | Sports and Outdoors | Toys and Games             | Video games   |

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
