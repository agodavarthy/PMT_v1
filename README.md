## Environment

Create the virtual environment by running:

pip -r requirements.txt

Download the Full Movielens data(58,000 movies) from this link: https://grouplens.org/datasets/movielens/latest/
Save it to data/ folder

## Preprocess
1) Matches the mentioned movies in redial data with the movielens IDs.

#### Redial:
python scripts/match_movies.py --redial=data/redial_dataset/movies_with_mentions.csv --ml_movies=data/ml-latest/movies.csv --output=data/redial/

This step is not needed for Gorecdial

2) Get the redial dataset and movielens dataset then use scripts/match_movies.py to create the matched up ids of movies, movie_match.csv.

#### Redial:
python scripts/reformat.py --movie_merged data/redial/movies_merged.csv --ml_movies data/ml-latest/movies.csv --output data/redial/movie_match.csv

#### Gorecdial:
python scripts/reformat_gorecdial.py --movie_merged data/gorecdial/movies_merged.csv --ml_movies data/ml-latest/movies.csv --output data/gorecdial/movie_match.csv

3) extract_movie_summary.py -- extracts the movie plots from my database

#### Redial:
python scripts/extract_movie_summary.py --movie_match data/redial/movie_match.csv --output data/redial/movie_plot.csv

#### Gorecdial:
python scripts/extract_movie_summary.py --movie_match data/gorecdial/movie_match.csv --output data/gorecdial/movie_plot.csv

4) split_movielens.py:  Split ratings data, keeping only movies we matched up and write out the
movie_map which includes the mapping to matrix_ids

inputs: movie_match.csv -- from match_movies.py
        ratings.csv -- from movielens dataset

outputs: movie_map.csv -- Entire mapping between movies, movielens and matrix ids
    train.npz, test.npz -- These are split ratings with matched movie items only 90/10 split

    cv/train.npz, cv/test.npz -- this train is 80% of ratings and test is 10%,
                                 essentially making it a 80/10/10 train/valid/test split


#### Redial:
python scripts/split_movielens.py -ratings data/ml-latest/ratings.csv -movie_match data/redial/movie_match.csv -o data/redial/

#### Gorecdial:
python scripts/split_movielens.py -ratings data/ml-latest/ratings.csv -movie_match data/gorecdial/movie_match.csv -o data/gorecdial/

## Pretrain
```bash
    run_pretrain.sh dataset model gpu

 Ex: bash run_pretrain.sh redial gmf 1
```
The file does the pretraining of nmf, mf or gmf model. 
Look at the parameters input to ```pretrain.py``` to make sure you pretrain on the correct data (redial or GoRecDial).

This file could also be used for parameter search. ```pretrain.py``` has 'for loops' in the file, that these loops are input
different combination of hyperparameters.

## Test
In the paper, we are experimenting transformer, bert, dan and elmo.
### Sentiment model
For each of the language models mention above, we need to train the sentiment model for later test phase.
For example, to train the sentiment model for elmo
```bash
run_sentiment.sh dataset elmo 0

Ex: bash run_sentiment.sh redial elmo 1
```
The argument 0 means to train the model on GPU 0. You need to rerun the scripts for other language models too.

### Test split
For each for the language models, run ```run_exp.sh``` to split the dataset.
```bash run_exp.sh dataset encoder gpu
Ex: bash run_exp.sh redial elmo 0
```
This example split the dataset with elmo.

### Test model
To get the results on dataset GoRecDial, run:
```bash
run_test_gorecdial.sh
```

To get the results on dataset Redial, run:
```bash
run_test_redial.sh
```
####
#Results are saved in eval_results folder. Metrics Can be calculated using aggregate_metrics.py. Of course you can also implement your own metric calculator
