# Therapeutic Companion Bot

## Introduction

In this project, we tried to build a chatbot using Natural Language Processing(NLP). 

## Collectioon

First, we scraped data from the following sources:

* Reddit - We scraped all the mental health subreddits containing the subreddit name, the questions asked, and all the comments added in the Response column.
* WHO (World Health Organization) — We scraped the ethical rules for creating a mental health chatbot here. This data is used to write guardrails for the models.
* HuggingFace - This data is a processed Question and Answer dataset between a psychiatrist and a patient.

* The data was around 5GB and in various formats, such as .csv, .parquet, .txt, and .json.

